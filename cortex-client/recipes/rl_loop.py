"""
RL against a colocated Neutrino training + sampling job.

A port of ``tinker_cookbook/recipes/math_rl/train.py``

Variable naming convention (see CONTRIBUTING.md in tinker-cookbook):
    _P: Problem dimension (different questions in a batch)
    _G: Group dimension (rollouts per problem, for reward centering)
    _D: Datum dimension (rollouts after flattening)
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import chz

from tinker_cookbook.utils import ml_log

from dss_client.neutrino_client import DEBUG_OPTIONS_ENV
from neutrino_common import (
    TrainSequence,
    build_renderer,
    collate,
    forward_backward_step,
    make_client,
    running_job,
    sequence_from_rollout,
    stop_params_for,
    sync_weights,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

# Match MathEnv / ProblemEnv defaults.
FORMAT_COEF = 0.1

_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class MathProblems:
    train: list[tuple[str, str]]
    test: list[tuple[str, str]] | None

    def __post_init__(self) -> None:
        if len(self.train) == 0:
            raise ValueError("a math dataset needs at least one training problem")


def load_math(seed: int = 0) -> MathProblems:
    """Hendrycks MATH train (MATH-500 held out) + HuggingFaceH4/MATH-500 test."""
    from tinker_cookbook.recipes.math_rl.math_env import (
        _get_hendrycks_math_test,
        _get_hendrycks_math_train,
    )
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    train_rows = _get_hendrycks_math_train().shuffle(seed=seed)
    train: list[tuple[str, str]] = []
    for row in train_rows:
        train.append((row["problem"], extract_boxed(row["solution"])))

    test_rows = _get_hendrycks_math_test()
    test: list[tuple[str, str]] = []
    for row in test_rows:
        test.append((row["problem"], extract_boxed(row["solution"])))

    return MathProblems(train=train, test=test or None)


def question_suffix() -> str:
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv

    return MathEnv.question_suffix()


def convo_prefix() -> list[dict[str, str]]:
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv

    return list(MathEnv.standard_fewshot_prefix())


def build_prompt(question: str, renderer) -> list[int]:
    conversation = [
        *convo_prefix(),
        {"role": "user", "content": question + question_suffix()},
    ]
    return renderer.build_generation_prompt(conversation).to_ints()


def _stopped_cleanly(result: dict, max_tokens: int | None) -> bool:
    finish_reason = result.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason != "length"
    if max_tokens is None:
        return True
    return len(result.get("token_ids") or []) < max_tokens


def score_response(
    response: str,
    answer: str,
    *,
    result: dict,
    max_tokens: int | None,
    format_coef: float = FORMAT_COEF,
) -> tuple[float, dict[str, float]]:
    from tinker_cookbook.recipes.math_rl.math_env import safe_grade
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    well_formed = _stopped_cleanly(result, max_tokens)
    try:
        given = extract_boxed(response)
        format_ok = True
    except ValueError:
        given = None
        format_ok = False

    correct_format = float(well_formed and format_ok)
    correct_answer = 0.0
    if format_ok and given is not None:
        correct_answer = float(safe_grade(given, answer))
    reward = format_coef * (correct_format - 1.0) + correct_answer
    return reward, {"format": correct_format, "correct": correct_answer}


@dataclass
class MathAccuracyEvaluator:
    prompts: list[list[int]]
    answers: list[str]
    sampling_params: dict = field(default_factory=dict)
    format_coef: float = FORMAT_COEF
    name: str = "test/env/all"

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.answers):
            raise ValueError(
                f"{len(self.prompts)} prompts but {len(self.answers)} answers"
            )

    def __call__(self, client: Any, job_id: str) -> dict[str, float]:
        if len(self.prompts) == 0:
            logger.warning("%s: no held-out problems, skipping", type(self).__name__)
            return {}

        request_id = client.generate(
            job_id, prompts=self.prompts, sampling_params=self.sampling_params
        )
        results = client.poll_request(job_id, request_id)["results"]
        if len(results) != len(self.prompts):
            raise RuntimeError(
                f"asked for {len(self.prompts)} completions, got {len(results)}"
            )

        max_tokens = self.sampling_params.get("max_tokens")
        corrects: list[float] = []
        formats: list[float] = []
        rewards: list[float] = []
        completion_lengths: list[int] = []
        n_truncated = 0

        for result, answer in zip(results, self.answers):
            text = result.get("text") or ""
            token_ids = result.get("token_ids") or []
            completion_lengths.append(len(token_ids))
            if not _stopped_cleanly(result, max_tokens):
                n_truncated += 1
            reward, metrics = score_response(
                text,
                answer,
                result=result,
                max_tokens=max_tokens,
                format_coef=self.format_coef,
            )
            corrects.append(metrics["correct"])
            formats.append(metrics["format"])
            rewards.append(reward)

        n = len(results)
        return {
            f"{self.name}/correct": sum(corrects) / n,
            f"{self.name}/format": sum(formats) / n,
            f"{self.name}/reward": sum(rewards) / n,
            f"{self.name}/frac_truncated": n_truncated / n,
            f"{self.name}/num_examples": float(n),
            f"{self.name}/mean_completion_tokens": sum(completion_lengths) / n,
        }


@chz.chz
class Config:
    config: str
    job_id: str | None = None

    model_name: str = "Qwen/Qwen3.5-4B"
    training_gpus: int = 4
    sampling_gpus: int = 4
    gpu_memory_utilization: float = 0.4
    micro_batch_size: int = 1
    zero_stage: int = 2
    attn_implementation: str = "flash_attention_3"
    dtype: str = "bfloat16"
    seed: int = 0
    max_seq_len: int = 8192

    problems_per_batch: int = 64
    group_size: int = 16
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    format_coef: float = FORMAT_COEF

    train_batch_size: int = 8
    max_tokens_per_mb: int = 10240
    learning_rate: float = 2e-5
    weight_decay: float = 0.0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    gradient_clipping: float | None = 1.0
    max_steps: int | None = None  # full epoch (~188 batches at batch=64)
    eps_clip: float = 0.2
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0
    remove_constant_reward_groups: bool = True

    lora_rank: int | None = None

    debug_image_tag: str | None = None

    # Evals. 0 disables; otherwise baseline at batch 0 and the final batch.
    eval_every: int = 20
    # MATH-500 is 500 problems — None means the full split.
    n_test: int | None = None
    # uses the training temperature / max_tokens by default.
    eval_temperature: float | None = None
    eval_max_tokens: int | None = None

    log_path: str = "/tmp/dss-examples/rl-loop"
    wandb_project: str = None
    wandb_name: str | None = None


def lora_config(config: Config) -> dict[str, Any] | None:
    if config.lora_rank is None:
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
    per_step = config.micro_batch_size * config.training_gpus
    if config.train_batch_size % per_step != 0:
        raise ValueError(
            f"train_batch_size ({config.train_batch_size}) must be a multiple of "
            f"micro_batch_size * training_gpus ({config.micro_batch_size} * "
            f"{config.training_gpus} = {per_step})"
        )

    training_config: dict = {
        "model_provider": "huggingface",
        "n_gpus": config.training_gpus,
        "max_seq_len": config.max_seq_len,
        "train_batch_size": config.train_batch_size,
        "attn_implementation": config.attn_implementation,
        "optimizer": {
            "name": "AdamW",
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": [config.adam_beta1, config.adam_beta2],
            "eps": config.adam_eps,
        },
        "mb_spec": {"max_tokens_per_mb": config.max_tokens_per_mb},
        "ds_config": {
            "train_batch_size": config.train_batch_size,
            "train_micro_batch_size_per_gpu": config.micro_batch_size,
            "gradient_accumulation_steps": config.train_batch_size // per_step,
            "zero_optimization": {"stage": config.zero_stage, "reduce_scatter": True},
            "bf16": {"enabled": True},
        },
    }
    peft = lora_config(config)
    if peft is not None:
        training_config["peft_config"] = peft
    if config.gradient_clipping is not None:
        training_config["gradient_clipping"] = config.gradient_clipping

    inference_config: dict = {
        "max_seq_len": config.max_seq_len,
        "n_gpus": config.sampling_gpus,
        "vllm_config": {
            "max_model_len": config.max_seq_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
        },
    }
    if peft is not None:
        inference_config["peft_config"] = peft

    body: dict = {
        "sub_job_configs": [
            {
                "job_type": "sampling",
                "model_name": config.model_name,
                "dtype": config.dtype,
                "seed": config.seed,
                "inference_config": inference_config,
            },
            {
                "job_type": "training",
                "model_name": config.model_name,
                "dtype": config.dtype,
                "seed": config.seed,
                "training_config": training_config,
            },
        ]
    }
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def processing_block(config: Config, global_batch_size: int) -> dict:
    return dict(
        loss_fn="grpo",
        config=dict(
            eps_clip=config.eps_clip,
            loss_agg_mode=config.loss_agg_mode,
            entropy_coeff=config.entropy_coeff,
            global_batch_size=global_batch_size,
        ),
    )


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

    _train(config, ml_logger)

    ml_logger.close()
    logger.info("Training completed")


def _train(config: Config, ml_logger: Any) -> None:
    tokenizer, renderer, renderer_name = build_renderer(config.model_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info("Using renderer: %s", renderer_name)

    logger.info("Loading MATH dataset...")
    math_dataset = load_math(seed=config.seed)
    train_problems = math_dataset.train

    n_train_batches = len(train_problems) // config.problems_per_batch
    total_steps = (
        n_train_batches
        if config.max_steps is None
        else min(n_train_batches, config.max_steps)
    )
    logger.info(
        "Training for %d rollout batches (%d problems, %d per batch, group_size=%d)",
        total_steps,
        len(train_problems),
        config.problems_per_batch,
        config.group_size,
    )

    stop_params = stop_params_for(renderer.get_stop_sequences())
    sampling_params = dict(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        **stop_params,
    )

    evaluator = None
    if config.eval_every > 0:
        if math_dataset.test is None:
            logger.warning(
                "eval_every=%d but MATH has no held-out split, so no benchmark will "
                "be reported",
                config.eval_every,
            )
        else:
            test_problems = math_dataset.test
            if config.n_test is not None:
                test_problems = test_problems[: config.n_test]
            eval_temperature = (
                config.temperature
                if config.eval_temperature is None
                else config.eval_temperature
            )
            evaluator = MathAccuracyEvaluator(
                prompts=[
                    build_prompt(question, renderer)
                    for question, _ in test_problems
                ],
                answers=[answer for _, answer in test_problems],
                sampling_params=dict(
                    max_tokens=config.eval_max_tokens or config.max_tokens,
                    temperature=eval_temperature,
                    **stop_params,
                ),
                format_coef=config.format_coef,
            )
            logger.info("Held-out benchmark on %d problems", len(evaluator.prompts))

    client = make_client(config.config)

    with running_job(
        client, job_body(config), job_id=config.job_id
    ) as job_id:
        for batch_idx in range(total_steps):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/batch": batch_idx,
                "progress/done_frac": (batch_idx + 1) / max(n_train_batches, 1),
                "optim/lr": config.learning_rate,
            }

            if evaluator is not None and _should_eval(
                batch_idx, total_steps, config.eval_every
            ):
                eval_start = time.time()
                metrics.update(evaluator(client, job_id))
                metrics["time/eval"] = time.time() - eval_start

            batch_start = batch_idx * config.problems_per_batch
            batch = train_problems[batch_start : batch_start + config.problems_per_batch]

            prompts_D: list[list[int]] = []
            prompt_tokens_P: list[list[int]] = []
            for question, _ in batch:
                prompt_tokens = build_prompt(question, renderer)
                prompt_tokens_P.append(prompt_tokens)
                prompts_D.extend([prompt_tokens] * config.group_size)

            request_id = client.generate(
                job_id, prompts=prompts_D, sampling_params=sampling_params
            )
            results_D = client.poll_request(job_id, request_id)["results"]
            if len(results_D) != len(prompts_D):
                raise RuntimeError(
                    f"asked for {len(prompts_D)} rollouts, got {len(results_D)} results"
                )

            rewards_P: list[float] = []
            corrects_P: list[float] = []
            formats_P: list[float] = []
            datums_D: list[TrainSequence] = []
            for problem_idx, (prompt_tokens, (_, answer)) in enumerate(
                zip(prompt_tokens_P, batch)
            ):
                group = results_D[
                    problem_idx * config.group_size : (problem_idx + 1) * config.group_size
                ]
                scored = [
                    score_response(
                        result.get("text") or "",
                        answer,
                        result=result,
                        max_tokens=config.max_tokens,
                        format_coef=config.format_coef,
                    )
                    for result in group
                ]
                rewards_G = [reward for reward, _ in scored]
                mean_reward = sum(rewards_G) / len(rewards_G)
                rewards_P.append(mean_reward)
                corrects_P.append(
                    sum(m["correct"] for _, m in scored) / len(scored)
                )
                formats_P.append(sum(m["format"] for _, m in scored) / len(scored))
                advantages_G = [reward - mean_reward for reward in rewards_G]

                if (
                    config.remove_constant_reward_groups
                    and all(advantage == 0.0 for advantage in advantages_G)
                ):
                    continue

                for result, advantage in zip(group, advantages_G):
                    sampled_tokens = [int(token) for token in (result.get("token_ids") or [])]
                    if len(sampled_tokens) == 0:
                        continue
                    datums_D.append(
                        sequence_from_rollout(
                            prompt_tokens,
                            sampled_tokens,
                            advantage=advantage,
                        )
                    )

            train_loss = float("nan")
            if len(datums_D) == 0:
                logger.warning(
                    "Batch %d: no rollouts to train on, skipping the optimizer step",
                    batch_idx,
                )
            else:
                kwargs, context = collate(
                    datums_D,
                    pad_token_id=pad_token_id,
                    max_seq_len=config.max_seq_len,
                    with_rl_context=True,
                )
                fwd_bwd_result, step_result = forward_backward_step(
                    client,
                    job_id,
                    kwargs,
                    context=context,
                    learning_rate=config.learning_rate,
                    processing=processing_block(config, global_batch_size=len(datums_D)),
                )
                train_loss = float(fwd_bwd_result["avg_loss"])
                metrics.update(fwd_bwd_result.get("metrics") or {})
                metrics.update(step_result.get("metrics") or {})
                sync_weights(
                    client,
                    job_id,
                    weight_format="lora" if config.lora_rank is not None else None,
                )

            metrics.update(
                {
                    "reward/mean": sum(rewards_P) / len(rewards_P),
                    "reward/std": (
                        statistics.pstdev(rewards_P) if len(rewards_P) > 1 else 0.0
                    ),
                    "env/all/correct": sum(corrects_P) / len(corrects_P),
                    "env/all/format": sum(formats_P) / len(formats_P),
                    "rollouts/total": len(results_D),
                    "rollouts/trained": len(datums_D),
                    "train/avg_loss": train_loss,
                    "time/total": time.time() - t_start,
                }
            )
            ml_logger.log_metrics(metrics, step=batch_idx)


def _should_eval(step: int, total_steps: int, eval_every: int) -> bool:
    return step % eval_every == 0 or step == total_steps - 1


if __name__ == "__main__":
    chz.nested_entrypoint(main)
