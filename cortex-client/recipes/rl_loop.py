"""
RL against a colocated Neutrino training + sampling job.

A port of ``tinker_cookbook/recipes/math_rl/train.py``

Variable naming convention (see CONTRIBUTING.md in tinker-cookbook):
    _P: Problem dimension (different questions in a batch)
    _G: Group dimension (rollouts per problem, for reward centering)
    _D: Datum dimension (rollouts after flattening)
"""

import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import chz

from tinker_cookbook import cli_utils
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

QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."

CONVO_PREFIX = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + QUESTION_SUFFIX,
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w "
            "6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}"
        ),
    },
]


@dataclass
class MathDataset:
    train: list[tuple[str, str]]
    test: list[tuple[str, str]] | None

    def __post_init__(self) -> None:
        if len(self.train) == 0:
            raise ValueError("a math dataset needs at least one training problem")


def load_gsm8k(
    dataset: str = "openai/gsm8k",
    dataset_config: str = "main",
    seed: int = 0,
    n_test: int = 200,
) -> MathDataset:
    import datasets

    loaded = datasets.load_dataset(dataset, dataset_config)
    assert isinstance(loaded, datasets.DatasetDict)

    train_rows = loaded["train"].shuffle(seed=seed)
    train = list(zip(train_rows["question"], train_rows["answer"]))

    if "test" not in loaded:
        logger.warning("%s has no test split, so no benchmark will be reported", dataset)
        return MathDataset(train=train, test=None)

    test_rows = loaded["test"].shuffle(seed=seed)
    test = list(zip(test_rows["question"], test_rows["answer"]))[:n_test]
    return MathDataset(train=train, test=test)


def build_prompt(question: str, renderer) -> list[int]:
    conversation = [
        *CONVO_PREFIX,
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    return renderer.build_generation_prompt(conversation).to_ints()


def get_reward(response: str, answer: str) -> float:
    from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


@dataclass
class GSM8KAccuracyEvaluator:
    prompts: list[list[int]]
    answers: list[str]
    sampling_params: dict = field(default_factory=dict)
    name: str = "test/gsm8k"

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.answers):
            raise ValueError(
                f"{len(self.prompts)} prompts but {len(self.answers)} answers"
            )

    def __call__(self, client: Any, job_id: str) -> dict[str, float]:
        from tinker_cookbook.eval.benchmarks._common import check_gsm8k
        from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer

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
        n_correct = n_wrong = n_truncated = n_errors = 0
        completion_lengths: list[int] = []

        for result, answer in zip(results, self.answers):
            text = result.get("text") or ""
            token_ids = result.get("token_ids") or []
            completion_lengths.append(len(token_ids))

            if _is_truncated(result, max_tokens):
                n_truncated += 1
                continue
            try:
                ground_truth = extract_gsm8k_final_answer(answer)
            except ValueError:
                n_errors += 1
                continue
            if check_gsm8k(text, ground_truth):
                n_correct += 1
            else:
                n_wrong += 1

        n = len(results)
        n_completed = n_correct + n_wrong
        return {
            f"{self.name}/accuracy": n_correct / n,
            f"{self.name}/score_completed": (
                n_correct / n_completed if n_completed > 0 else float("nan")
            ),
            f"{self.name}/frac_truncated": n_truncated / n,
            f"{self.name}/frac_errors": n_errors / n,
            f"{self.name}/num_examples": float(n),
            f"{self.name}/mean_completion_tokens": sum(completion_lengths) / n,
        }


def _is_truncated(result: dict, max_tokens: int | None) -> bool:
    finish_reason = result.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason == "length"
    if max_tokens is None:
        return False
    return len(result.get("token_ids") or []) >= max_tokens


@chz.chz
class Config:
    config: str
    job_id: str | None = None
    keep_job: bool | None = None

    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5"
    training_gpus: int = 4
    sampling_gpus: int = 4
    gpu_memory_utilization: float = 0.4
    micro_batch_size: int = 1
    zero_stage: int = 2
    attn_implementation: str = "flash_attention_3"
    dtype: str = "bfloat16"
    seed: int = 42
    max_seq_len: int = 8192

    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    problems_per_batch: int = 16
    group_size: int = 8
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0

    train_batch_size: int = 8
    max_tokens_per_mb: int = 10240  # backend microbatch token budget
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gradient_clipping: float | None = 1.0
    max_steps: int = 100
    eps_clip: float = 0.2
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0

    lora_rank: int = 8
    # Evaluation. 0 disables it; otherwise a baseline runs at batch 0 and the
    # final batch is always evaluated.
    eval_every: int = 10
    n_test: int = 64
    eval_temperature: float = 0.6
    eval_max_tokens: int | None = None

    log_path: str = "/tmp/neutrino-examples/rl-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    debug_image_tag: str | None = None


# Qwen-style Linear names covering Tinker's train_attn / train_mlp / train_unembed.
_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def lora_config(config: Config) -> dict:
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

    peft = lora_config(config)
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
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "mb_spec": {"max_tokens_per_mb": config.max_tokens_per_mb},
        "ds_config": {
            "train_batch_size": config.train_batch_size,
            "train_micro_batch_size_per_gpu": config.micro_batch_size,
            "gradient_accumulation_steps": config.train_batch_size // per_step,
            "zero_optimization": {"stage": config.zero_stage, "reduce_scatter": True},
            "bf16": {"enabled": True},
        },
        "peft_config": peft,
    }
    if config.gradient_clipping is not None:
        training_config["gradient_clipping"] = config.gradient_clipping

    body: dict = {
        "sub_job_configs": [
            {
                "job_type": "sampling",
                "model_name": config.model_name,
                "dtype": config.dtype,
                "seed": config.seed,
                "inference_config": {
                    "max_seq_len": config.max_seq_len,
                    "n_gpus": config.sampling_gpus,
                    "vllm_config": {
                        "max_model_len": config.max_seq_len,
                        "gpu_memory_utilization": config.gpu_memory_utilization,
                    },
                    "peft_config": peft,
                },
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
        # Client refuses a create-job `debug` block unless this is set.
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
        config.model_name, config.renderer_name
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading dataset...")
    math_dataset = load_gsm8k(
        config.dataset, config.dataset_config, seed=0, n_test=config.n_test
    )
    train_problems = math_dataset.train

    n_train_batches = len(train_problems) // config.problems_per_batch
    total_steps = (
        n_train_batches if config.max_steps is None else min(n_train_batches, config.max_steps)
    )
    logger.info(f"Training for {total_steps} rollout batches")

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
                "eval_every=%d but %s has no held-out split, so no benchmark will "
                "be reported",
                config.eval_every,
                config.dataset,
            )
        else:
            evaluator = GSM8KAccuracyEvaluator(
                prompts=[build_prompt(question, renderer) for question, _ in math_dataset.test],
                answers=[answer for _, answer in math_dataset.test],
                sampling_params=dict(
                    max_tokens=config.eval_max_tokens or config.max_tokens,
                    temperature=config.eval_temperature,
                    **stop_params,
                ),
            )
            logger.info(f"Held-out benchmark on {len(evaluator.prompts)} problems")

    client = make_client(config.config)

    with running_job(
        client, job_body(config), job_id=config.job_id, keep_job=config.keep_job
    ) as job_id:
        for batch_idx in range(total_steps):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/batch": batch_idx,
                "progress/done_frac": (batch_idx + 1) / n_train_batches,
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

            request_id = client.generate(
                job_id, prompts=prompts_D, sampling_params=sampling_params
            )
            results_D = client.poll_request(job_id, request_id)["results"]
            if len(results_D) != len(prompts_D):
                raise RuntimeError(
                    f"asked for {len(prompts_D)} rollouts, got {len(results_D)} results"
                )

            rewards_P: list[float] = []
            datums_D: list[TrainSequence] = []
            for problem_idx, (prompt_tokens, (_, answer)) in enumerate(
                zip(prompt_tokens_P, batch)
            ):
                group = results_D[
                    problem_idx * config.group_size : (problem_idx + 1) * config.group_size
                ]
                rewards_G = [get_reward(result.get("text", ""), answer) for result in group]
                mean_reward = sum(rewards_G) / len(rewards_G)
                rewards_P.append(mean_reward)
                advantages_G = [reward - mean_reward for reward in rewards_G]

                if all(advantage == 0.0 for advantage in advantages_G):
                    continue

                for result, advantage in zip(group, advantages_G):
                    sampled_tokens = [int(token) for token in result.get("token_ids")]
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
                    "Batch %d: no rollout carried a non-zero advantage, skipping "
                    "the optimizer step",
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
                sync_weights(client, job_id, weight_format="lora")

            metrics.update(
                {
                    "reward/mean": sum(rewards_P) / len(rewards_P),
                    "reward/std": statistics.pstdev(rewards_P) if len(rewards_P) > 1 else 0.0,
                    "rollouts/total": len(results_D),
                    "rollouts/trained": len(datums_D),
                    "train/avg_loss": train_loss,
                    "time/total": time.time() - t_start,
                }
            )
            ml_logger.log_metrics(metrics, step=batch_idx)

    ml_logger.close()
    logger.info("Training completed")


def _should_eval(step: int, total_steps: int, eval_every: int) -> bool:
    return step % eval_every == 0 or step == total_steps - 1


if __name__ == "__main__":
    chz.nested_entrypoint(main)
