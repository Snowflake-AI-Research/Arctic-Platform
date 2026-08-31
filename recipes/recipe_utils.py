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
"""Shared pieces of the SFT and RL recipes: connection, rendering, collation, ops.

TODO(resync-recipes): ported from `cortex-client/recipes/`, which the
`jaelee/doc-refresh` branch restructures into `cortex_training/recipes/` (new
layout, YAML recipe configs, PEFT + checkpoint/inference helpers). Re-sync these
two recipes against that layout once it merges.

TODO(onprem-recipes): these recipes drive the remote Cortex backend only. The
client API is backend-agnostic, but `fwd_bwd`'s *batch* is not: Cortex takes the
RPC-style ``{"args", "kwargs", "context"}`` body built here, while on-prem takes
a pre-tokenized verl-GRPO ``{"batch", "meta", "processing"}``. Adding an on-prem
path means a second packager here plus an `OnPremConfig` branch in
`client_config`; it is a follow-up, not a hidden branch in these files.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from arctic_platform.client import ArcticClient
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import SamplingConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.client.cortex import connection

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_backend(config_path: str | None = None) -> CortexConfig:
    """Resolve the Cortex connection the same way the `cortex` CLI does.

    Without a path this falls back to ``CORTEX_CONFIG``, whatever ``cortex login``
    remembered, and then ``CORTEX_*`` / ``SNOWFLAKE_*`` env, so a recipe can run
    with no ``--config`` flag once you are logged in.
    """
    return connection.resolve(config_path=config_path)


def lora_config(rank: int) -> dict[str, Any] | None:
    """A LoRA adapter config with ``alpha == r``; None for dense fine-tuning."""
    if rank <= 0:
        return None
    return {
        "peft_type": "Lora",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(LORA_TARGET_MODULES),
    }


def reconnect_ids(job_id: str | None, *roles: str) -> dict[str, str]:
    """``ArcticClientConfig`` job-id kwargs addressing an existing job's sub-jobs.

    Assumes the recipe's own one-sub-job-per-role layout so this stays a pure
    function. `CortexJobs.attach` reads the real sub-jobs off the server instead,
    which is what you want for a job you did not create.
    """
    if job_id is None:
        return {}
    return {f"{role}_job_id": f"{job_id}:{role}:0" for role in roles}


@contextlib.contextmanager
def running_client(config: ArcticClientConfig, client_cls: type[ArcticClient]) -> Iterator[ArcticClient]:
    """Yield a connected client, releasing the GPUs of jobs *this* run created.

    Constructing the client creates the jobs and waits for them to run. A config
    carrying job ids attached to someone else's job, so it is left running.
    """
    attached = config.training_job_id is not None
    client = client_cls(config)
    logger.info("training job %s is running", client.jobs.training)
    try:
        yield client
    finally:
        if attached:
            logger.info("leaving pre-existing job %s running", client.jobs.training)
        else:
            client.shutdown()


def build_renderer(model_name: str):
    """Return ``(tokenizer, renderer, renderer_name)`` for ``model_name``.

    Uses tinker ``model_info.get_recommended_renderer_name``. These recipe
    helpers only cover models tinker lists; the server itself can host more.
    """
    from tinker_cookbook import model_info
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    try:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    except KeyError as exc:
        raise ValueError(
            f"tinker_cookbook has no recommended renderer for {model_name!r}; "
            "see https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/#available-renderers"
        ) from exc
    return tokenizer, renderers.get_renderer(renderer_name, tokenizer), renderer_name


def stop_params_for(stop_sequences: Sequence[Any]) -> dict:
    token_ids = [int(stop) for stop in stop_sequences if isinstance(stop, int) and not isinstance(stop, bool)]
    strings = [stop for stop in stop_sequences if isinstance(stop, str)]

    params: dict = {}
    if len(token_ids) > 0:
        params["stop_token_ids"] = token_ids
    if len(strings) > 0:
        params["stop"] = strings
    return params


@dataclass
class TrainSequence:
    input_ids: list[int]
    labels: list[int]
    advantage: float = 0.0

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                f"input_ids ({len(self.input_ids)}) and labels ({len(self.labels)}) must have the same length"
            )


def sequence_from_conversation(
    messages: Sequence[Any],
    renderer: Any,
    train_on_what: Any,
    max_seq_len: int | None = None,
) -> TrainSequence:
    """Render a chat conversation straight into the server's forward-backward shape.

    ``renderer.build_supervised_example`` tokenizes the whole conversation and
    returns per-token weights aligned with those tokens: ``weights[i] > 0`` marks
    token ``i`` as one the model should learn to produce. It covers every
    assistant turn in one sequence, which is the reason for using a renderer at
    all -- ``apply_chat_template(return_assistant_tokens_mask=True)`` only works
    for templates carrying ``{% generation %}`` markers, and Qwen3's does not
    (HF then returns an all-zero mask).
    """
    model_input, weights = renderer.build_supervised_example(list(messages), train_on_what=train_on_what)
    token_ids = [int(token) for token in model_input.to_ints()]
    token_weights = [float(weight) for weight in weights.tolist()]
    if len(token_ids) != len(token_weights):
        raise ValueError(f"renderer returned {len(token_ids)} tokens but {len(token_weights)} weights")

    if max_seq_len is not None:
        token_ids = token_ids[:max_seq_len]
        token_weights = token_weights[:max_seq_len]
    if len(token_ids) < 2:
        raise ValueError("need at least 2 tokens to build a training sequence")

    labels = [IGNORE_INDEX] * len(token_ids)
    for position in range(len(token_ids) - 1):
        if token_weights[position + 1] > 0.0:
            labels[position] = token_ids[position + 1]
    return TrainSequence(input_ids=token_ids, labels=labels)


def sequence_from_rollout(
    prompt_tokens: Sequence[int],
    sampled_tokens: Sequence[int],
    advantage: float = 0.0,
) -> TrainSequence:
    if len(prompt_tokens) == 0:
        raise ValueError("prompt_tokens must be non-empty")
    if len(sampled_tokens) == 0:
        raise ValueError("sampled_tokens must be non-empty")

    tokens = [int(token) for token in prompt_tokens] + [int(token) for token in sampled_tokens]
    n_prompt = len(prompt_tokens)
    labels = [IGNORE_INDEX] * len(tokens)
    for offset in range(len(sampled_tokens)):
        position = n_prompt - 1 + offset
        labels[position] = tokens[position + 1]
    return TrainSequence(input_ids=tokens, labels=labels, advantage=advantage)


def collate(
    sequences: Sequence[TrainSequence],
    pad_token_id: int,
    max_seq_len: int,
    pad_to_max_seq_len: bool = False,
    with_rl_context: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if len(sequences) == 0:
        raise ValueError("collate needs at least one sequence")

    longest = max(len(sequence.input_ids) for sequence in sequences)
    if longest > max_seq_len:
        raise ValueError(
            f"a sequence is {longest} tokens but max_seq_len is {max_seq_len}; "
            "truncate while rendering, or line the training and sampling "
            "max_seq_len up with each other"
        )

    width = max_seq_len if pad_to_max_seq_len else longest

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    advantages: list[list[float]] = []
    loss_mask: list[list[float]] = []

    for sequence in sequences:
        padding = width - len(sequence.input_ids)
        input_ids.append(sequence.input_ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(sequence.input_ids) + [0] * padding)
        padded_labels = sequence.labels + [IGNORE_INDEX] * padding
        labels.append(padded_labels)
        if with_rl_context:
            mask = [1.0 if label != IGNORE_INDEX else 0.0 for label in padded_labels]
            loss_mask.append(mask)
            advantages.append([sequence.advantage * m for m in mask])

    kwargs: dict[str, Any] = dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        position_ids=torch.arange(width, dtype=torch.long).expand(len(sequences), -1).contiguous(),
        use_cache=False,
    )
    context: dict[str, torch.Tensor] = {}
    if with_rl_context:
        context = dict(
            input_ids=kwargs["input_ids"],
            advantages=torch.tensor(advantages, dtype=torch.float32),
            loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
        )
    else:
        kwargs["labels"] = torch.tensor(labels, dtype=torch.long)

    return kwargs, context


def client_config(
    backend: CortexConfig,
    model_name: str,
    max_seq_len: int,
    seed: int,
    dtype: str,
    training_gpus: int,
    lora_rank: int,
    ds_config: dict[str, Any],
    ds_worker_config: dict[str, Any],
    sampling_gpus: int = 0,
    gpu_memory_utilization: float | None = None,
    job_ready_timeout: float = 3600.0,
    job_id: str | None = None,
) -> ArcticClientConfig:
    """Assemble the one config both recipes hand to `running_client`.

    ``to_cortex()`` lifts the Cortex-typed fields out of ``ds_config`` /
    ``ds_worker_config``; the sampling sub-job is created only when
    ``sampling_gpus > 0``.
    """
    roles = ("training", "sampling") if sampling_gpus > 0 else ("training",)
    vllm = {} if gpu_memory_utilization is None else {"gpu_memory_utilization": gpu_memory_utilization}
    return ArcticClientConfig(
        model_name=model_name,
        seed=seed,
        dtype=dtype,
        max_seq_len=max_seq_len,
        training_gpus=training_gpus,
        sampling_gpus=sampling_gpus,
        job_ready_timeout=job_ready_timeout,
        backend=backend,
        training=TrainingConfig(
            peft=lora_config(lora_rank),
            ds_config=ds_config,
            ds_worker_config=ds_worker_config,
        ),
        sampling=SamplingConfig(vllm=vllm),
        **reconnect_ids(job_id, *roles),
    )


def train_step(
    client: ArcticClient,
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    learning_rate: float | None = None,
    processing: dict | None = None,
) -> tuple[dict, dict]:
    """One ``fwd_bwd`` + ``step``, in the Cortex RPC batch shape.

    Not `ArcticSFTClient.train_step`: that one injects on-prem's
    ``processing={"loss_fn": "sft"}`` contract. See the TODO(sft-client) in
    `sft_loop.py`, which this helper goes away with.
    """
    batch: dict[str, Any] = {"args": [], "kwargs": kwargs}
    if context:
        batch["context"] = context
    fwd_bwd_result = client.fwd_bwd(batch, processing=processing)
    step_result = client.step(learning_rate=learning_rate)
    return fwd_bwd_result, step_result
