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

"""MATH-500 eval against an inference endpoint."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import chz
from recipes.utils import build_renderer
from recipes.utils import make_client
from recipes.utils import stop_params_for
from recipes.inference.endpoint import generate_results
from recipes.inference.endpoint import inference_endpoint_body
from recipes.inference.endpoint import running_inference_endpoint

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)


@dataclass
class _MathExample:
    prompt_text: str
    answer: str
    example_id: str


def _load_math500(max_examples: int | None = None) -> list[_MathExample]:
    from datasets import load_dataset
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples: list[_MathExample] = []
    for idx, row in enumerate(ds):
        try:
            expected = extract_boxed(row["solution"])
        except ValueError:
            continue
        examples.append(
            _MathExample(
                prompt_text=row["problem"],
                answer=expected,
                example_id=f"math500-{idx}",
            )
        )
    return examples


def _run_math500(
    *,
    client: Any,
    job_id: str,
    renderer: Any,
    max_examples: int | None,
    sampling_params: dict[str, Any],
    generate_batch_size: int,
    max_seq_len: int,
) -> dict[str, float]:
    from recipes.rl.math_grpo.train import build_prompt
    from recipes.rl.math_grpo.train import score_response

    examples = _load_math500(max_examples)
    if len(examples) == 0:
        raise ValueError("MATH-500 produced no examples")
    prompts: list[list[int]] = []
    for example in examples:
        tokens = build_prompt(example.prompt_text, renderer)
        if len(tokens) >= max_seq_len:
            raise ValueError(
                f"{example.example_id} prompt has {len(tokens)} tokens; raise max_seq_len (currently {max_seq_len})"
            )
        prompts.append(tokens)

    results = generate_results(client, job_id, prompts, sampling_params, generate_batch_size)
    n_correct = 0
    format_sum = 0.0
    max_tokens = sampling_params.get("max_tokens")
    for example, result in zip(examples, results):
        text = result.get("text") or ""
        _reward, metrics = score_response(
            text,
            example.answer,
            result=result,
            max_tokens=max_tokens,
        )
        n_correct += int(metrics["correct"] >= 1.0)
        format_sum += float(metrics.get("format") or 0.0)
    n = len(examples)
    score = n_correct / n
    logger.info("math500: %.1f%% (%d/%d)", 100.0 * score, n_correct, n)
    metrics = {
        "math500/correct": score,
        "math500/format": format_sum / n,
        "math500/num_examples": float(n),
        "test/env/all/correct": score,
        "test/env/all/format": format_sum / n,
        "test/env/all/num_examples": float(n),
    }
    logger.info("Results: %s", metrics)
    return metrics


@chz.chz
class Config:
    config: str
    job_id: str | None = None  # attach to a running inference endpoint

    model_name: str = "Qwen/Qwen3-8B"
    n_gpus: int = 2
    max_seq_len: int = 8192
    gpu_memory_utilization: float = 0.8
    dtype: str = "bfloat16"
    seed: int = 0
    lora_rank: int = 0
    debug_image_tag: str | None = None
    keep_job: bool | None = None

    source_job_id: str | None = None
    # Required with source_job_id. Use the cp_* id.
    checkpoint_id: str | None = None

    max_examples: int | None = None
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    generate_batch_size: int = 64


def run_evaluation(
    *,
    config_path: str,
    model_name: str,
    source_job_id: str | None = None,
    checkpoint_id: str | None = None,
    job_id: str | None = None,
    lora_rank: int = 0,
    n_gpus: int = 2,
    max_seq_len: int = 8192,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "bfloat16",
    seed: int = 0,
    max_examples: int | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generate_batch_size: int = 64,
    debug_image_tag: str | None = None,
    keep_job: bool | None = None,
) -> dict[str, float]:
    if debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
        logger.info("Using debug image_tag=%s", debug_image_tag)

    _, renderer, renderer_name = build_renderer(model_name)
    logger.info("Using renderer: %s", renderer_name)

    client = make_client(config_path)
    body, source = inference_endpoint_body(
        model_name=model_name,
        max_seq_len=max_seq_len,
        n_gpus=n_gpus,
        dtype=dtype,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_rank=lora_rank,
        source_job_id=source_job_id,
        checkpoint_id=checkpoint_id,
        debug_image_tag=debug_image_tag,
    )
    if source is not None:
        logger.info(
            "Starting MATH-500 eval from weights-only checkpoint %s (job %s)",
            source["checkpoint_id"],
            source["source_job_id"],
        )
    elif job_id is None:
        logger.info("Starting MATH-500 eval from original weights (%s)", model_name)

    sampling_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        **stop_params_for(renderer.get_stop_sequences()),
    }

    with running_inference_endpoint(
        client,
        body,
        job_id=job_id,
        keep_job=keep_job,
        lora_rank=lora_rank,
    ) as eval_job_id:
        return _run_math500(
            client=client,
            job_id=eval_job_id,
            renderer=renderer,
            max_examples=max_examples,
            sampling_params=sampling_params,
            generate_batch_size=generate_batch_size,
            max_seq_len=max_seq_len,
        )


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_evaluation(
        config_path=config.config,
        model_name=config.model_name,
        source_job_id=config.source_job_id,
        checkpoint_id=config.checkpoint_id,
        job_id=config.job_id,
        lora_rank=config.lora_rank,
        n_gpus=config.n_gpus,
        max_seq_len=config.max_seq_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        dtype=config.dtype,
        seed=config.seed,
        max_examples=config.max_examples,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        generate_batch_size=config.generate_batch_size,
        debug_image_tag=config.debug_image_tag,
        keep_job=config.keep_job,
    )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
