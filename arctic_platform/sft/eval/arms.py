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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched evaluation arms for the packed SFT integration plan.

Values come from docs/ap_integration_plan.md section 3b, user answers (liger on,
tiled_mlp token_chunk_size 4096, FlashAttention 3, UltraChat), and code defaults
for every other knob (ActivationOffloadPatch, TiledRematPatch.token_budget).
"""

from __future__ import annotations

from typing import Any

# Plan section 3b.
LR = 1e-5
SEED = 42
MICRO_BATCH = 1
ATTN = "flash_attention_3"  # plan named FA2; user required FA3 on H200
LIGER = True
TILED_MLP_TOKEN_CHUNK_SIZE = 4096  # user answer
TILED_REMAT_TOKEN_BUDGET = 4096  # TiledRematPatch default / plan budget
WARMUP_STEPS = 1
PROBE_STEPS = 3  # plan: 3 steps after 1 warmup

# Plan section 5 AT 1-GPU walls (tokens), for control and acceptance.
AT_WALLS = {
    "Qwen/Qwen3-8B": {"gc": 262_144, "tiling": 278_528, "offload": 606_208, "tiling_offload": 868_352},
    "Qwen/Qwen3.5-4B": {"gc": 344_064, "tiling": 425_984, "offload": 524_288, "tiling_offload": 819_200},
    "Qwen/Qwen3.5-9B": {"gc": 237_568, "tiling": 278_528, "offload": 442_368, "tiling_offload": 704_512},
}


def ds_config(*, world_size: int = 1) -> dict[str, Any]:
    """ZeRO-2 with optimizer states on CPU. deepspeed_engine maps this to CPU Adam."""
    return {
        "train_micro_batch_size_per_gpu": MICRO_BATCH,
        "train_batch_size": world_size * MICRO_BATCH,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "none"},
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        },
    }


def ds_worker_config(arm: str) -> dict[str, Any]:
    """Flat worker knobs for ModelSpec.from_ds_worker_config.

    Arms: gc, tiling, offload, tiling_offload. tiled_mlp is on in every arm (plan 3b).
    """
    if arm not in {"gc", "tiling", "offload", "tiling_offload"}:
        raise ValueError(f"unknown arm {arm!r}")
    cfg: dict[str, Any] = {
        "attn_implementation": ATTN,
        "use_liger": LIGER,
        "enable_gradient_checkpointing": True,
        "tiled_mlp_token_chunk_size": TILED_MLP_TOKEN_CHUNK_SIZE,
    }
    if arm in {"tiling", "tiling_offload"}:
        cfg["tiled_remat_token_budget"] = TILED_REMAT_TOKEN_BUDGET
    if arm in {"offload", "tiling_offload"}:
        cfg["enable_activation_offload"] = True
    return cfg


def job_config(model_name: str, arm: str, *, world_size: int = 1) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "job_type": "sft",
        "seed": SEED,
        "full_determinism": False,
        "ds_config": ds_config(world_size=world_size),
        "ds_worker_config": ds_worker_config(arm),
        "training_config": {
            "optimizer": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "gradient_accumulation_steps": 1,
        },
    }
