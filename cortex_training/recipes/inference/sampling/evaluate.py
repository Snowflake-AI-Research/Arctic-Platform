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
MATH-500 sampling eval.
"""

from __future__ import annotations

import logging
import os

import chz
from recipes._shared.cortex_training import build_renderer
from recipes._shared.cortex_training import inference_job_body
from recipes._shared.cortex_training import make_client
from recipes._shared.cortex_training import prepare_inference_weights
from recipes._shared.cortex_training import running_job
from recipes._shared.cortex_training import source_checkpoint_info
from recipes._shared.cortex_training import stop_params_for
from recipes.inference.sampling.benchmarks import run_math500

from cortex_training.client import DEBUG_OPTIONS_ENV

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)


@chz.chz
class Config:
    config: str
    job_id: str | None = None  # attach to running sampling job

    model_name: str = "Qwen/Qwen3-8B"
    n_gpus: int = 2
    training_gpus: int | None = None
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
    training_gpus: int | None = None,
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
    source = source_checkpoint_info(
        source_job_id,
        checkpoint_id,
    )
    if source is not None:
        logger.info(
            "Starting MATH-500 eval from weights-only checkpoint %s (job %s)",
            source["checkpoint_id"],
            source["source_job_id"],
        )
    elif job_id is None:
        if lora_rank > 0:
            raise ValueError(
                "lora_rank is unused for original-weight eval; omit it. "
                "LoRA adapters load from a weights-only checkpoint via source_job_id"
            )
        logger.info("Starting MATH-500 eval from original weights (%s)", model_name)

    body = inference_job_body(
        model_name=model_name,
        max_seq_len=max_seq_len,
        n_gpus=n_gpus,
        dtype=dtype,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_rank=lora_rank,
        source_checkpoint_info=source,
        training_gpus=training_gpus,
        debug_image_tag=debug_image_tag,
    )
    sampling_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        **stop_params_for(renderer.get_stop_sequences()),
    }

    attached = job_id is not None
    with running_job(client, body, job_id=job_id, keep_job=keep_job) as eval_job_id:
        if not attached:
            prepare_inference_weights(client, eval_job_id, body, lora_rank=lora_rank)
        result = run_math500(
            client=client,
            job_id=eval_job_id,
            renderer=renderer,
            max_examples=max_examples,
            sampling_params=sampling_params,
            generate_batch_size=generate_batch_size,
            max_seq_len=max_seq_len,
        )
        logger.info(
            "%s: %.1f%% (%d/%d)",
            result.name,
            100.0 * result.score,
            result.num_correct,
            result.num_examples,
        )
        logger.info("Results: %s", result.metrics)
        return result.metrics


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
        training_gpus=config.training_gpus,
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
