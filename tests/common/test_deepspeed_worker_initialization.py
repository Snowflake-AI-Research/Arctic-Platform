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

"""Canonical dense model and engine specs used by the on-prem worker."""

import pytest

pytest.importorskip("ray")

from arctic_platform.common.deepspeed_worker import _dense_model_settings
from arctic_platform.common.deepspeed_worker import _training_engine_spec
from arctic_platform.model import build_deepspeed_config
from arctic_platform.model import dense_model_spec


def test_onprem_dense_adapter_preserves_model_patches():
    worker = {
        "model_provider": "liger",
        "attn_implementation": "flash_attention_3",
        "enable_gradient_checkpointing": False,
        "compile": {"fullgraph": False},
        "tiled_mlp_token_chunk_size": 64,
        "fp32_lm_head": True,
        "fused_lm_head_token_chunk_size": 128,
        "fused_lm_head_vocab_chunk_size": 4096,
        "peft_config": {"peft_type": "Lora", "r": 8, "target_modules": ["q_proj"]},
    }

    settings = _dense_model_settings("model", {"dtype": "bfloat16"}, worker, 1)
    spec = dense_model_spec(settings)

    assert spec.model_path_or_name == "model"
    assert spec.patches.liger is True
    assert spec.patches.gradient_checkpointing is False
    assert spec.patches.compile.fullgraph is False
    assert spec.patches.tiled_mlp.token_chunk_size == 64
    assert spec.patches.lm_head.fp32 is True
    assert spec.patches.lm_head.token_chunk_size == 128
    assert spec.patches.peft["r"] == 8


def test_onprem_engine_adapter_translates_optimizer_and_preserves_scheduler_and_gas():
    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 3,
        "gradient_clipping": 0.5,
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}},
        "optimizer": {"type": "AdamW", "params": {"lr": 3e-4, "torch_adam": True}},
        "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 10}},
    }

    spec = _training_engine_spec({"dtype": "bfloat16", "seed": 7}, ds_config, 4, 1)
    resolved = build_deepspeed_config(spec)

    assert spec.optimizer.name == "torch_adamw"
    assert spec.optimizer.lr == 3e-4
    assert spec.optimizer.fused is False
    assert resolved["train_batch_size"] == 24
    assert resolved["zero_optimization"] == ds_config["zero_optimization"]
    assert "optimizer" not in resolved
    assert resolved["scheduler"] == ds_config["scheduler"]
    assert resolved["seed"] == 7


def test_onprem_worker_keeps_historical_gradient_checkpointing_default():
    settings = _dense_model_settings(
        "model",
        {"dtype": "bfloat16"},
        {"attn_implementation": "eager"},
        1,
    )

    assert settings.patches.gradient_checkpointing is True


@pytest.mark.parametrize(
    "worker,sp_size,match",
    [
        ({"attn_implementation": "eager", "ep_size": 2}, 1, "expert-parallel"),
        ({"attn_implementation": "eager"}, 2, "sequence-parallel"),
    ],
)
def test_onprem_distributed_followups_fail_before_model_load(worker, sp_size, match):
    with pytest.raises(ValueError, match=match):
        _dense_model_settings("model", {"dtype": "bfloat16"}, worker, sp_size)
