"""
Minimal supervised fine-tuning loop against a Neutrino training job.

A port of ``tinker_cookbook/recipes/sl_loop.py``.

This sample script uses tinker_cookbook's util functions-- supports models tinker supports.
"""

import logging
import time

import chz
import datasets

from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from neutrino_common import (
    build_renderer,
    collate,
    forward_backward_step,
    make_client,
    running_job,
    sequence_from_conversation,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)


@chz.chz
class Config:
    config: str
    job_id: str | None = None
    keep_job: bool | None = None

    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str  = "qwen3"
    n_gpus: int = 8
    micro_batch_size: int = 1
    zero_stage: int = 2
    attn_implementation: str = "flash_attention_3"
    dtype: str = "bfloat16"
    seed: int = 42

    dataset: str = "HuggingFaceH4/no_robots"
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_length: int = 2048
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    pad_to_max_length: bool = False
    max_steps: int = 20

    log_path: str = "/tmp/neutrino-examples/sft-loop"
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

    return {
        "sub_job_configs": [
            {
                "job_type": "training",
                "model_name": config.model_name,
                "dtype": config.dtype,
                "seed": config.seed,
                "training_config": {
                    "model_provider": "huggingface",
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
                },
            }
        ]
    }


def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer, renderer, renderer_name = build_renderer(
        config.model_name, config.renderer_name
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading dataset...")
    dataset = datasets.load_dataset(config.dataset)
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"].shuffle(seed=0)

    n_train_batches = len(train_dataset) // config.batch_size
    n_dropped = len(train_dataset) % config.batch_size
    if n_dropped:
        logger.info(
            f"Dropping last {n_dropped} examples to keep batch size uniform at "
            f"{config.batch_size}"
        )
    total_steps = (
        n_train_batches if config.max_steps is None else min(n_train_batches, config.max_steps)
    )
    logger.info(f"Train batches: {n_train_batches}; training for {total_steps} steps")

    client = make_client(config.config)

    with running_job(
        client, job_body(config), job_id=config.job_id, keep_job=config.keep_job
    ) as job_id:
        for step in range(total_steps):
            start_time = time.time()
            metrics: dict[str, float] = {}

            # Linear learning rate schedule, applied on the server per step.
            lr_mult = max(0.0, 1.0 - step / n_train_batches)
            current_lr = config.learning_rate * lr_mult

            batch_start = step * config.batch_size
            batch_rows = train_dataset.select(
                range(batch_start, batch_start + config.batch_size)
            )
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
            fwd_bwd_result, step_result = forward_backward_step(
                client, job_id, kwargs, learning_rate=current_lr
            )

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
