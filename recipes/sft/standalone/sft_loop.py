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
"""Minimal supervised fine-tuning loop against a remote Cortex training job.

A port of ``tinker_cookbook/recipes/sl_loop.py``, driven by the unified client.

    python -m recipes.sft.standalone.sft_loop config=recipes/config.json
"""

from __future__ import annotations

import logging
import time
from typing import Any

import chz
import datasets
from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from arctic_platform.client import ArcticClient
from recipes.recipe_utils import build_renderer
from recipes.recipe_utils import client_config
from recipes.recipe_utils import collate
from recipes.recipe_utils import load_backend
from recipes.recipe_utils import running_client
from recipes.recipe_utils import sequence_from_conversation
from recipes.recipe_utils import train_step

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)


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

    log_path: str = "/tmp/arctic-recipes/sft-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None


def build_config(config: Config):
    per_step = config.micro_batch_size * config.n_gpus
    if config.batch_size % per_step != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be a multiple of "
            f"micro_batch_size * n_gpus ({config.micro_batch_size} * {config.n_gpus} = {per_step})"
        )
    if config.ep_size is not None:
        if config.ep_size <= 0:
            raise ValueError(f"ep_size must be positive, got {config.ep_size}")
        if config.n_gpus % config.ep_size != 0:
            raise ValueError(f"n_gpus ({config.n_gpus}) must be a multiple of ep_size ({config.ep_size})")

    worker: dict[str, Any] = {
        "attn_implementation": config.attn_implementation,
        "model_provider": config.model_provider,
    }
    if config.ep_size is not None:
        worker["ep_size"] = config.ep_size

    return client_config(
        backend=load_backend(config.config),
        model_name=config.model_name,
        max_seq_len=config.max_length,
        seed=config.seed,
        dtype=config.dtype,
        training_gpus=config.n_gpus,
        lora_rank=config.lora_rank,
        ds_config={
            "train_batch_size": config.batch_size,
            "train_micro_batch_size_per_gpu": config.micro_batch_size,
            "gradient_accumulation_steps": config.batch_size // per_step,
            "zero_optimization": {"stage": config.zero_stage},
            "bf16": {"enabled": True},
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
        },
        ds_worker_config=worker,
        job_id=config.job_id,
    )


def main(config: Config):
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

    # TODO(sft-client): switch to ArcticSFTClient (and common.train_step to its
    # train_step) once a live Cortex deployment is confirmed to register the `sft`
    # loss_fn. Plain ArcticClient for now: Cortex derives the SFT loss from the
    # `labels` in the batch and takes no processing block, while ArcticSFTClient
    # would inject on-prem's `processing={"loss_fn": "sft"}` — and loss_fn names
    # resolve against the deployed backend's registry, not a schema we can check
    # here. Switching is a one-line change plus dropping the local train_step.
    with running_client(build_config(config), ArcticClient) as client:
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
            fwd_bwd_result, step_result = train_step(client, kwargs, learning_rate=current_lr)

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
