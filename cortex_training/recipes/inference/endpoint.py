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

"""Create and call a Cortex Training inference endpoint.

An inference endpoint is a running job that serves generations. On the wire the
worker is still ``job_type=sampling`` — the same generation runtime RL uses for
rollouts. This helper is the standalone serving path, without a training
sub-job.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from recipes.utils import prepare_inference_weights
from recipes.utils import running_job
from recipes.utils import sampling_job_body
from recipes.utils import source_checkpoint_info

from cortex_training import CortexTrainingClient

logger = logging.getLogger(__name__)


def inference_endpoint_body(
    *,
    model_name: str,
    max_seq_len: int,
    n_gpus: int,
    dtype: str = "bfloat16",
    seed: int = 42,
    gpu_memory_utilization: float = 0.8,
    lora_rank: int = 0,
    source_job_id: str | None = None,
    checkpoint_id: str | None = None,
    debug_image_tag: str | None = None,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Return ``(create-job body, source checkpoint info or None)``."""
    source = source_checkpoint_info(source_job_id, checkpoint_id)
    body = sampling_job_body(
        model_name=model_name,
        max_seq_len=max_seq_len,
        n_gpus=n_gpus,
        dtype=dtype,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_rank=lora_rank,
        source_checkpoint_info=source,
        debug_image_tag=debug_image_tag,
    )
    return body, source


@contextlib.contextmanager
def running_inference_endpoint(
    client: CortexTrainingClient,
    body: dict[str, Any],
    *,
    job_id: str | None = None,
    keep_job: bool | None = None,
    lora_rank: int = 0,
) -> Iterator[str]:
    """Yield a running inference endpoint id.

    Creates the job from ``body`` unless ``job_id`` is set. ``keep_job`` follows
    :func:`running_job`: attach keeps the endpoint, a job this helper created is
    cancelled on exit, unless overridden.
    """
    attached = job_id is not None
    with running_job(client, body, job_id=job_id, keep_job=keep_job) as endpoint_id:
        if not attached:
            prepare_inference_weights(client, endpoint_id, body, lora_rank=lora_rank)
        yield endpoint_id


def generate_results(
    client: Any,
    job_id: str,
    prompts: list[list[int]],
    sampling_params: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Generate completions for tokenized prompts on a running endpoint.

    ``sampling_params`` is the generate-request field name on the Cortex Training
    API (temperature, max_tokens, and related decoding settings).
    """
    results: list[dict[str, Any]] = []
    width = max(1, batch_size)
    for start in range(0, len(prompts), width):
        batch = prompts[start : start + width]
        request_id = client.generate(job_id, prompts=batch, sampling_params=sampling_params)
        payload = client.poll_request(job_id, request_id)
        batch_results = payload.get("results") or []
        if len(batch_results) != len(batch):
            raise RuntimeError(f"asked for {len(batch)} completions, got {len(batch_results)}")
        for result in batch_results:
            results.append(result if isinstance(result, dict) else {"text": str(result)})
    return results


def log_endpoint_ready(config_path: str, job_id: str) -> None:
    logger.info("Inference endpoint is ready: job_id=%s", job_id)
    logger.info(
        "Examples against this endpoint:\n"
        "  python -m recipes.inference.generate config=%s job_id=%s\n"
        "  python -m recipes.inference.evaluate config=%s job_id=%s",
        config_path,
        job_id,
        config_path,
        job_id,
    )
    logger.info("Tear down with: cortex-training cancel %s", job_id)
