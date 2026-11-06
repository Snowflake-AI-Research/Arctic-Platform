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
Submit a text prompt and print the completion.
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
from recipes.inference.sampling.benchmarks import generate_results
from recipes.inference.sampling.generate import completion_text
from recipes.inference.sampling.generate import render_user_prompt

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
    max_seq_len: int = 4096
    gpu_memory_utilization: float = 0.8
    dtype: str = "bfloat16"
    seed: int = 42
    lora_rank: int = 0
    debug_image_tag: str | None = None
    keep_job: bool | None = None

    source_job_id: str | None = None
    # Required with source_job_id. Use the cp_* id.
    checkpoint_id: str | None = None
    # False (default): same disable-thinking renderer as conversational SFT.
    # True: thinking-on renderer. Must match the training run. renderer_name overrides.
    enable_thinking: bool = False
    renderer_name: str | None = None

    prompt: str = "How many r's are in strawberry?"
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 1.0


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"

    tokenizer, renderer, renderer_name = build_renderer(
        config.model_name,
        renderer_name=config.renderer_name,
        enable_thinking=config.enable_thinking,
    )
    logger.info(
        "Using renderer: %s (enable_thinking=%s)",
        renderer_name,
        config.enable_thinking,
    )

    if config.job_id is None and config.source_job_id is None and config.lora_rank > 0:
        raise ValueError(
            "lora_rank is unused for original-weight sampling; omit it. "
            "LoRA adapters load from a weights-only checkpoint via source_job_id"
        )

    client = make_client(config.config)
    source = source_checkpoint_info(
        config.source_job_id,
        config.checkpoint_id,
    )
    if source is not None:
        logger.info(
            "Starting sampling from weights-only checkpoint %s (job %s)",
            source["checkpoint_id"],
            source["source_job_id"],
        )

    body = inference_job_body(
        model_name=config.model_name,
        max_seq_len=config.max_seq_len,
        n_gpus=config.n_gpus,
        dtype=config.dtype,
        seed=config.seed,
        gpu_memory_utilization=config.gpu_memory_utilization,
        lora_rank=config.lora_rank,
        source_checkpoint_info=source,
        training_gpus=config.training_gpus,
        debug_image_tag=config.debug_image_tag,
    )
    sampling_params = {
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        **stop_params_for(renderer.get_stop_sequences()),
    }
    prompt_tokens = render_user_prompt(renderer, config.prompt)
    attached = config.job_id is not None
    with running_job(client, body, job_id=config.job_id, keep_job=config.keep_job) as job_id:
        if not attached:
            prepare_inference_weights(client, job_id, body, lora_rank=config.lora_rank)
        results = generate_results(client, job_id, [prompt_tokens], sampling_params, batch_size=1)
        raw = completion_text(results[0])
        logger.info("Prompt: %s", config.prompt)
        logger.info(
            "Renderer: %s (enable_thinking=%s)",
            renderer_name,
            config.enable_thinking,
        )
        logger.info("Completion:\n%s", raw)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
