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

A port of ``tinker_cookbook/recipes/chat_sl/train.py``.

Default data is a one-example chat dataset that memorizes
``Who trained you?`` → ``Snowflake AI Research``. Hugging Face chat datasets with a
``messages`` column work as well. This sample script uses tinker_cookbook's
util functions-- supports models tinker supports.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import chz
import datasets
from recipes._shared.cortex_training import build_renderer
from recipes._shared.cortex_training import collate
from recipes._shared.cortex_training import forward_backward_step
from recipes._shared.cortex_training import forward_loss
from recipes._shared.cortex_training import log_saved_checkpoints
from recipes._shared.cortex_training import lora_peft_config
from recipes._shared.cortex_training import make_client
from recipes._shared.cortex_training import running_job
from recipes._shared.cortex_training import save_recipe_checkpoints
from recipes._shared.cortex_training import sequence_from_conversation
from recipes._shared.cortex_training import use_next_token_labels
from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_RECIPE_DIR = Path(__file__).resolve().parent
BUILTIN_CHAT_DATASETS = {
    "who_trained_you": _RECIPE_DIR / "data" / "who_trained_you.jsonl",
}
WHO_TRAINED_YOU_PROMPT = "Who trained you?"


def is_who_trained_you_dataset(dataset: str) -> bool:
    return dataset == "who_trained_you" or Path(dataset).name == "who_trained_you.jsonl"


@chz.chz
class Config:
    config: str
    job_id: str | None = None

    model_name: str = "Qwen/Qwen3-8B"
    n_gpus: int = 4
    micro_batch_size: int = 1
    zero_stage: int = 2
    attn_implementation: str = "flash_attention_3"
    dtype: str = "bfloat16"
    seed: int = 42
    # huggingface for dense / LoRA; prime_rl for MoE with expert parallelism.
    model_provider: str = "huggingface"
    ep_size: int | None = None

    dataset: str = "who_trained_you"
    dataset_split: str = "train"
    test_split: str | None = "test"
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_length: int = 2048
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    pad_to_max_length: bool = False
    max_steps: int = 100

    # 0 = dense FT. Set e.g. 32 for LoRA (r == alpha).
    lora_rank: int = 0
    eval_every: int = 20
    debug_image_tag: str | None = None
    # False (default): tinker *_disable_thinking renderer when the model has one.
    # True: thinking-on renderer (qwen3 for Qwen3-8B).
    enable_thinking: bool = False
    renderer_name: str | None = None

    log_path: str = "/tmp/cortex-training-examples/sft-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None


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
    peft = lora_peft_config(config.lora_rank)
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


def resolve_chat_dataset(dataset: str) -> str:
    builtin = BUILTIN_CHAT_DATASETS.get(dataset)
    if builtin is not None:
        return str(builtin)
    return dataset


def _is_local_chat_file(source: str) -> bool:
    path = Path(source).expanduser()
    return path.is_file() and path.suffix.lower() in {".json", ".jsonl"}


def tile_rows(dataset: datasets.Dataset, n_rows: int) -> datasets.Dataset:
    """Repeat a short dataset so training can run ``max_steps`` batches."""
    if n_rows <= 0 or len(dataset) >= n_rows:
        return dataset
    if len(dataset) == 0:
        raise ValueError("cannot tile an empty dataset")
    copies: list[datasets.Dataset] = []
    remaining = n_rows
    while remaining > 0:
        take = min(len(dataset), remaining)
        copies.append(dataset.select(range(take)))
        remaining -= take
    return datasets.concatenate_datasets(copies)


def load_chat_dataset(
    dataset: str,
    *,
    dataset_split: str,
    test_split: str | None,
    n_train: int,
    n_test: int,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    source = resolve_chat_dataset(dataset)
    if _is_local_chat_file(source):
        data_files: dict[str, str] = {dataset_split: source}
        if test_split:
            data_files[test_split] = source
        loaded = datasets.load_dataset("json", data_files=data_files)
    else:
        loaded = datasets.load_dataset(source)
    if not isinstance(loaded, datasets.DatasetDict):
        loaded = datasets.DatasetDict({dataset_split: loaded})

    train_dataset = tile_rows(loaded[dataset_split], n_train).shuffle(seed=0)
    test_dataset = None
    if test_split:
        if test_split in loaded:
            test_dataset = tile_rows(loaded[test_split], n_test)
        else:
            logger.info(
                "%s has no %s split; skipping test/nll (available: %s)",
                dataset,
                test_split,
                list(loaded),
            )
    return train_dataset, test_dataset


def eval_nll(
    client,
    job_id: str,
    test_dataset,
    renderer,
    train_on_what: renderers.TrainOnWhat,
    *,
    pad_token_id: int,
    max_length: int,
    batch_size: int,
    pad_to_max_length: bool,
    next_token_labels: bool = False,
) -> dict[str, float]:
    """Mean CE loss on the held-out split via forward-backward (no optimizer step)."""
    if test_dataset is None or len(test_dataset) < batch_size:
        return {}
    n_batches = len(test_dataset) // batch_size
    losses: list[float] = []
    n_sequences = 0
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        rows = test_dataset.select(range(start, start + batch_size))
        sequences = [
            sequence_from_conversation(
                row["messages"],
                renderer,
                train_on_what=train_on_what,
                max_seq_len=max_length,
                next_token_labels=next_token_labels,
            )
            for row in rows
        ]
        kwargs, _ = collate(
            sequences,
            pad_token_id=pad_token_id,
            max_seq_len=max_length,
            pad_to_max_seq_len=pad_to_max_length,
        )
        result = forward_loss(client, job_id, kwargs)
        losses.append(float(result["avg_loss"]))
        n_sequences += len(sequences)
    return {
        "test/nll": sum(losses) / len(losses),
        "test/num_examples": float(n_sequences),
    }


def _should_eval(step: int, every: int, total_steps: int) -> bool:
    return every > 0 and (step % every == 0 or step == total_steps - 1)


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

    tokenizer, renderer, renderer_name = build_renderer(
        config.model_name,
        renderer_name=config.renderer_name,
        enable_thinking=config.enable_thinking,
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info(
        "Using renderer: %s (enable_thinking=%s)",
        renderer_name,
        config.enable_thinking,
    )
    next_token_labels = use_next_token_labels(config.model_provider)

    logger.info("Loading dataset...")
    train_dataset, test_dataset = load_chat_dataset(
        config.dataset,
        dataset_split=config.dataset_split,
        test_split=config.test_split,
        n_train=config.max_steps * config.batch_size,
        n_test=config.batch_size,
    )

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

            if _should_eval(step, config.eval_every, total_steps):
                metrics.update(
                    eval_nll(
                        client,
                        job_id,
                        test_dataset,
                        renderer,
                        config.train_on_what,
                        pad_token_id=pad_token_id,
                        max_length=config.max_length,
                        batch_size=config.batch_size,
                        pad_to_max_length=config.pad_to_max_length,
                        next_token_labels=next_token_labels,
                    )
                )

            # Linear learning rate schedule, applied on the server per step.
            lr_mult = max(0.0, 1.0 - step / max(n_train_batches, 1))
            current_lr = config.learning_rate * lr_mult

            batch_start = step * config.batch_size
            batch_rows = train_dataset.select(range(batch_start, batch_start + config.batch_size))
            sequences = [
                sequence_from_conversation(
                    row["messages"],
                    renderer,
                    train_on_what=config.train_on_what,
                    max_seq_len=config.max_length,
                    next_token_labels=next_token_labels,
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

        saved = save_recipe_checkpoints(client, job_id)
        sample_prompt = WHO_TRAINED_YOU_PROMPT if is_who_trained_you_dataset(config.dataset) else None
        log_saved_checkpoints(
            config_path=config.config,
            job_id=job_id,
            saved=saved,
            sampling_command="sample",
            lora_rank=config.lora_rank,
            sample_prompt=sample_prompt,
            enable_thinking=config.enable_thinking,
            renderer_name=config.renderer_name,
            model_name=config.model_name,
            n_gpus=config.n_gpus,
            temperature=0 if sample_prompt else None,
        )

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
