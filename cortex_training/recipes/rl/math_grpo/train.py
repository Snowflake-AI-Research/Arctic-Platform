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
RL against a colocated Cortex Training training + sampling job.

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
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import chz
from recipes._shared.cortex_training import TrainSequence
from recipes._shared.cortex_training import bootstrap_router_replay
from recipes._shared.cortex_training import build_renderer
from recipes._shared.cortex_training import collate
from recipes._shared.cortex_training import discard_router_replay
from recipes._shared.cortex_training import forward_backward_step
from recipes._shared.cortex_training import log_saved_checkpoints
from recipes._shared.cortex_training import lora_peft_config
from recipes._shared.cortex_training import make_client
from recipes._shared.cortex_training import router_replay_config
from recipes._shared.cortex_training import router_replay_stop_params
from recipes._shared.cortex_training import running_job
from recipes._shared.cortex_training import sampling_params_with_sample_ids
from recipes._shared.cortex_training import sampling_sub_job_id
from recipes._shared.cortex_training import save_recipe_checkpoints
from recipes._shared.cortex_training import sequence_from_rollout
from recipes._shared.cortex_training import stop_params_for
from recipes._shared.cortex_training import sync_weights
from tinker_cookbook.utils import ml_log

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

# Match MathEnv / ProblemEnv defaults.
FORMAT_COEF = 0.1


@dataclass
class MathProblems:
    train: list[tuple[str, str]]
    test: list[tuple[str, str]] | None

    def __post_init__(self) -> None:
        if len(self.train) == 0:
            raise ValueError("a math dataset needs at least one training problem")


def load_math(seed: int = 0) -> MathProblems:
    """Hendrycks MATH train (MATH-500 held out) + HuggingFaceH4/MATH-500 test."""
    from tinker_cookbook.recipes.math_rl.math_env import _get_hendrycks_math_test
    from tinker_cookbook.recipes.math_rl.math_env import _get_hendrycks_math_train
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    train_rows = _get_hendrycks_math_train().shuffle(seed=seed)
    train: list[tuple[str, str]] = []
    for row in train_rows:
        train.append((row["problem"], extract_boxed(row["solution"])))

    test: list[tuple[str, str]] = []
    for row in _get_hendrycks_math_test():
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
            raise ValueError(f"{len(self.prompts)} prompts but {len(self.answers)} answers")

    def __call__(self, client: Any, job_id: str) -> dict[str, float]:
        if len(self.prompts) == 0:
            logger.warning("%s: no held-out problems, skipping", type(self).__name__)
            return {}

        request_id = client.generate(job_id, prompts=self.prompts, sampling_params=self.sampling_params)
        results = client.poll_request(job_id, request_id)["results"]
        if len(results) != len(self.prompts):
            raise RuntimeError(f"asked for {len(self.prompts)} completions, got {len(results)}")

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

    model_name: str = "Qwen/Qwen3-8B"
    training_gpus: int = 4
    sampling_gpus: int = 4
    gpu_memory_utilization: float = 0.4
    micro_batch_size: int = 1
    zero_stage: int = 2
    attn_implementation: str = "flash_attention_3"
    dtype: str = "bfloat16"
    seed: int = 0
    max_seq_len: int = 8192
    # huggingface for dense / LoRA; prime_rl for MoE with expert parallelism.
    model_provider: str = "huggingface"
    ep_size: int | None = None

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

    # 0 = dense FT. Set e.g. 32 for LoRA (r == alpha).
    lora_rank: int = 32

    router_replay: bool = False
    router_replay_max_cache_bytes: int | None = None

    debug_image_tag: str | None = None

    # Evals. 0 disables; otherwise baseline at batch 0, every N, and the last batch.
    eval_every: int = 10
    # Caps sampling.evaluate max_examples. None runs the full split.
    n_test: int | None = None
    eval_temperature: float | None = None
    eval_max_tokens: int | None = None

    log_path: str = "/tmp/cortex-training-examples/rl-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None


def job_body(config: Config) -> dict:
    per_step = config.micro_batch_size * config.training_gpus
    if config.train_batch_size % per_step != 0:
        raise ValueError(
            f"train_batch_size ({config.train_batch_size}) must be a multiple of "
            f"micro_batch_size * training_gpus ({config.micro_batch_size} * "
            f"{config.training_gpus} = {per_step})"
        )
    if config.ep_size is not None:
        if config.ep_size <= 0:
            raise ValueError(f"ep_size must be positive, got {config.ep_size}")
        if config.training_gpus % config.ep_size != 0:
            raise ValueError(
                f"training_gpus ({config.training_gpus}) must be a multiple of ep_size ({config.ep_size})"
            )

    training_config: dict = {
        "model_provider": config.model_provider,
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
    if config.ep_size is not None:
        training_config["ep_size"] = config.ep_size
    if config.model_provider == "prime_rl" and config.router_replay:
        training_config["prime_rl"] = {
            "fused_cross_entropy": False,
        }
    peft = lora_peft_config(config.lora_rank)
    if peft is not None:
        training_config["peft_config"] = peft
    if config.gradient_clipping is not None:
        training_config["gradient_clipping"] = config.gradient_clipping
    if config.router_replay:
        training_config["router_replay"] = router_replay_config(
            max_cache_bytes=config.router_replay_max_cache_bytes,
        )

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
    if config.router_replay:
        inference_config["router_replay"] = router_replay_config(
            max_cache_bytes=config.router_replay_max_cache_bytes,
        )

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


def _should_eval(step: int, total_steps: int, eval_every: int) -> bool:
    return eval_every > 0 and (step % eval_every == 0 or step == total_steps - 1)


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
    total_steps = n_train_batches if config.max_steps is None else min(n_train_batches, config.max_steps)
    logger.info(
        "Training for %d rollout batches (%d problems, %d per batch, group_size=%d)",
        total_steps,
        len(train_problems),
        config.problems_per_batch,
        config.group_size,
    )

    stop_params = (
        router_replay_stop_params(renderer.get_stop_sequences(), tokenizer)
        if config.router_replay
        else stop_params_for(renderer.get_stop_sequences())
    )
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
                "eval_every=%d but MATH has no held-out split, so no benchmark will be reported",
                config.eval_every,
            )
        else:
            test_problems = math_dataset.test
            if config.n_test is not None:
                test_problems = test_problems[: config.n_test]
            eval_temperature = config.temperature if config.eval_temperature is None else config.eval_temperature
            evaluator = MathAccuracyEvaluator(
                prompts=[build_prompt(question, renderer) for question, _ in test_problems],
                answers=[answer for _, answer in test_problems],
                sampling_params=dict(
                    max_tokens=config.eval_max_tokens or config.max_tokens,
                    temperature=eval_temperature,
                    top_p=config.top_p,
                    **stop_params,
                ),
                format_coef=config.format_coef,
            )
            logger.info("Held-out MATH-500 on %d problems", len(evaluator.prompts))
        logger.info("After save, also run sampling.evaluate (MATH-500)")

    client = make_client(config.config)

    with running_job(client, job_body(config), job_id=config.job_id) as job_id:
        sampling_job_id: str | None = None
        if config.router_replay:
            logger.info("Bootstrapping router replay for job %s", job_id)
            bootstrap_router_replay(
                client,
                job_id,
                max_cache_bytes=config.router_replay_max_cache_bytes,
            )
            sampling_job_id = sampling_sub_job_id(job_id)

        for batch_idx in range(total_steps):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/batch": batch_idx,
                "progress/done_frac": (batch_idx + 1) / max(n_train_batches, 1),
                "optim/lr": config.learning_rate,
            }

            if evaluator is not None and _should_eval(batch_idx, total_steps, config.eval_every):
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

            sample_ids_D = [f"rl-{batch_idx}-{rollout_idx}" for rollout_idx in range(len(prompts_D))]
            generate_params: dict | list[dict] = sampling_params
            if config.router_replay:
                generate_params = sampling_params_with_sample_ids(sampling_params, sample_ids_D)

            request_id = client.generate(job_id, prompts=prompts_D, sampling_params=generate_params)
            results_D = client.poll_request(job_id, request_id)["results"]
            if len(results_D) != len(prompts_D):
                raise RuntimeError(f"asked for {len(prompts_D)} rollouts, got {len(results_D)} results")

            rewards_P: list[float] = []
            corrects_P: list[float] = []
            formats_P: list[float] = []
            datums_D: list[TrainSequence] = []
            trained_sample_ids: list[str] = []
            for problem_idx, (prompt_tokens, (_, answer)) in enumerate(zip(prompt_tokens_P, batch)):
                group_slice = slice(
                    problem_idx * config.group_size,
                    (problem_idx + 1) * config.group_size,
                )
                group = results_D[group_slice]
                group_sample_ids = sample_ids_D[group_slice]
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
                corrects_P.append(sum(m["correct"] for _, m in scored) / len(scored))
                formats_P.append(sum(m["format"] for _, m in scored) / len(scored))
                advantages_G = [reward - mean_reward for reward in rewards_G]

                if config.remove_constant_reward_groups and all(advantage == 0.0 for advantage in advantages_G):
                    continue

                for result, advantage, sample_id in zip(group, advantages_G, group_sample_ids):
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
                    trained_sample_ids.append(sample_id)

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
                    temperature=config.temperature,
                )
                fwd_bwd_result, step_result = forward_backward_step(
                    client,
                    job_id,
                    kwargs,
                    context=context,
                    learning_rate=config.learning_rate,
                    processing=processing_block(config, global_batch_size=len(datums_D)),
                    rr_sample_ids=(trained_sample_ids if config.router_replay else None),
                    router_replay_sampling_job_id=sampling_job_id,
                )
                train_loss = float(fwd_bwd_result["avg_loss"])
                metrics.update(fwd_bwd_result.get("metrics") or {})
                metrics.update(step_result.get("metrics") or {})
                sync_weights(
                    client,
                    job_id,
                    weight_format="lora" if config.lora_rank > 0 else None,
                )

            if config.router_replay:
                trained_id_set = set(trained_sample_ids)
                unused_sample_ids = [sample_id for sample_id in sample_ids_D if sample_id not in trained_id_set]
                if unused_sample_ids:
                    discard_router_replay(client, job_id, unused_sample_ids)

            metrics.update(
                {
                    "reward/mean": sum(rewards_P) / len(rewards_P),
                    "reward/std": statistics.pstdev(rewards_P) if len(rewards_P) > 1 else 0.0,
                    "env/all/correct": sum(corrects_P) / len(corrects_P),
                    "env/all/format": sum(formats_P) / len(formats_P),
                    "rollouts/total": len(results_D),
                    "rollouts/trained": len(datums_D),
                    "train/avg_loss": train_loss,
                    "time/total": time.time() - t_start,
                }
            )
            ml_logger.log_metrics(metrics, step=batch_idx)

        saved = save_recipe_checkpoints(client, job_id)
        log_saved_checkpoints(
            config_path=config.config,
            job_id=job_id,
            saved=saved,
            lora_rank=config.lora_rank,
            sampling_command="evaluate",
            model_name=config.model_name,
            n_gpus=config.sampling_gpus,
            temperature=(config.temperature if config.eval_temperature is None else config.eval_temperature),
            max_tokens=config.eval_max_tokens or config.max_tokens,
            top_p=config.top_p,
            seed=config.seed,
            max_examples=config.n_test,
        )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
