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

"""
Minimal supervised fine-tuning loop against a Cortex Training training job.

A port of ``tinker_cookbook/recipes/sl_loop.py``.

This sample script uses tinker_cookbook's util functions-- supports models tinker supports.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import chz
import datasets
from recipes._shared.cortex_training import build_renderer
from recipes._shared.cortex_training import collate
from recipes._shared.cortex_training import forward_backward_step
from recipes._shared.cortex_training import make_client
from recipes._shared.cortex_training import running_job
from recipes._shared.cortex_training import sequence_from_conversation
from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@chz.chz
class Config:
    config: str
    job_id: str | None = None

    model_name: str = "Qwen/Qwen3-8B"
    n_gpus: int = 8
    micro_batch_size: int = 1
    zero_stage: int = 2
    attn_implementation: str = "flash_attention_3"
    dtype: str = "bfloat16"
    seed: int = 42
    # huggingface for dense / LoRA; prime_rl for MoE with expert parallelism.
    model_provider: str = "huggingface"
    ep_size: int | None = None

    dataset: str = "HuggingFaceH4/no_robots"
    dataset_split: str = "train"
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_length: int = 2048
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    pad_to_max_length: bool = False
    max_steps: int = 100

    # 0 = dense FT. Set e.g. 32 for LoRA (r == alpha).
    lora_rank: int = 32
    debug_image_tag: str | None = None

    log_path: str = "/tmp/cortex-training-examples/sft-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None


def lora_config(config: Config) -> dict[str, Any] | None:
    if config.lora_rank <= 0:
        return None
    return {
        "peft_type": "Lora",
        "r": config.lora_rank,
        "lora_alpha": config.lora_rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(_LORA_TARGET_MODULES),
    }


def job_body(config: Config) -> dict:
    per_step = config.micro_batch_size * config.n_gpus
    if config.batch_size % per_step != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be a multiple of "
            f"micro_batch_size * n_gpus ({config.micro_batch_size} * "
            f"{config.n_gpus} = {per_step})"
        )
    if config.ep_size is not None:
        if config.ep_size <= 0:
            raise ValueError(f"ep_size must be positive, got {config.ep_size}")
        if config.n_gpus % config.ep_size != 0:
            raise ValueError(f"n_gpus ({config.n_gpus}) must be a multiple of ep_size ({config.ep_size})")

    training_config: dict[str, Any] = {
        "model_provider": config.model_provider,
        "n_gpus": config.n_gpus,
        "max_seq_len": config.max_length,
        "train_batch_size": config.batch_size,
        "attn_implementation": config.attn_implementation,
        "optimizer": {
            "name": "AdamW",
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "ds_config": {
            "train_batch_size": config.batch_size,
            "train_micro_batch_size_per_gpu": config.micro_batch_size,
            "gradient_accumulation_steps": config.batch_size // per_step,
            "zero_optimization": {"stage": config.zero_stage},
            "bf16": {"enabled": True},
        },
    }
    if config.ep_size is not None:
        training_config["ep_size"] = config.ep_size
    peft = lora_config(config)
    if peft is not None:
        training_config["peft_config"] = peft

    body: dict[str, Any] = {
        "sub_job_configs": [
            {
                "job_type": "training",
                "model_name": config.model_name,
                "dtype": config.dtype,
                "seed": config.seed,
                "training_config": training_config,
            }
        ]
    }
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def main(config: Config):
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", config.debug_image_tag)

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer, renderer, renderer_name = build_renderer(config.model_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading dataset...")
    dataset = datasets.load_dataset(config.dataset)
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset[config.dataset_split].shuffle(seed=0)

    n_train_batches = len(train_dataset) // config.batch_size
    n_dropped = len(train_dataset) % config.batch_size
    if n_dropped:
        logger.info(f"Dropping last {n_dropped} examples to keep batch size uniform at {config.batch_size}")
    total_steps = min(n_train_batches, config.max_steps)
    logger.info(f"Train batches: {n_train_batches}; training for {total_steps} steps")

    client = make_client(config.config)

    with running_job(client, job_body(config), job_id=config.job_id) as job_id:
        for step in range(total_steps):
            start_time = time.time()
            metrics: dict[str, float] = {}

            # Linear learning rate schedule, applied on the server per step.
            lr_mult = max(0.0, 1.0 - step / n_train_batches)
            current_lr = config.learning_rate * lr_mult

            batch_start = step * config.batch_size
            batch_rows = train_dataset.select(range(batch_start, batch_start + config.batch_size))
            sequences = [
                sequence_from_conversation(
                    row["messages"],
                    renderer,
                    train_on_what=config.train_on_what,
                    max_seq_len=config.max_length,
                )
                for row in batch_rows
            ]
            kwargs, _ = collate(
                sequences,
                pad_token_id=pad_token_id,
                max_seq_len=config.max_length,
                pad_to_max_seq_len=config.pad_to_max_length,
            )
            fwd_bwd_result, step_result = forward_backward_step(client, job_id, kwargs, learning_rate=current_lr)

            train_loss = float(fwd_bwd_result["avg_loss"])
            metrics.update(fwd_bwd_result.get("metrics") or {})
            metrics.update(step_result.get("metrics") or {})

            metrics.update(
                train_mean_nll=train_loss,
                global_steps=step_result.get("global_steps", step + 1),
                progress=step / n_train_batches,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=step)

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
