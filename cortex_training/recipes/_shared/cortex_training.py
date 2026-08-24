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

"""Shared Cortex Training recipe helpers."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cortex_training import CortexTrainingClient
from cortex_training import wire

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
_ROUTER_REPLAY_DEFAULT_MAX_CACHE_BYTES = 16 * 1024**3
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def make_client(config_path: str, **overrides: Any) -> CortexTrainingClient:
    parsed = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    config = parsed.get("connection", parsed)

    pat = config.get("pat")
    kwargs: dict[str, Any] = dict(
        database=config.get("database", "CORTEX_TRAINING_DB"),
        schema=config.get("schema", "PUBLIC"),
        endpoint=config.get("endpoint", "cortex-training"),
        poll_interval=float(config.get("poll_interval", 0.5)),
        poll_timeout=float(config.get("poll_timeout", 1800.0)),
    )
    kwargs.update(overrides)

    host = config.get("host")
    if host is None:
        raise ValueError("connection config needs `host` (Snowflake PAT auth)")
    if pat is None:
        raise ValueError("no PAT found: put `pat` in the connection config")
    return CortexTrainingClient.from_pat(
        host=host,
        pat=pat,
        verify_ssl=bool(config.get("verify_ssl", True)),
        **kwargs,
    )


def training_sub_job_id(job_id: str) -> str:
    return f"{job_id}:training:0"


def sampling_sub_job_id(job_id: str) -> str:
    return f"{job_id}:sampling:0"


def lora_peft_config(rank: int) -> dict[str, Any] | None:
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


def checkpoint_id_from_stage_path(stage_path: str | None) -> str | None:
    """Return the checkpoint id (``cp_*``) from a save-poll ``stage_path``.
    Example: ``s3://.../checkpoints/cp_<uuid>/global_step1/`` → ``cp_<uuid>``.
    """
    if not stage_path:
        return None
    parts = str(stage_path).split("/")
    for part in parts:
        if part.startswith("cp_"):
            return part
    return None


def save_checkpoint(
    client: CortexTrainingClient,
    job_id: str,
    *,
    checkpoint_type: str = "weights-only",
) -> dict[str, Any]:
    request_id = client.save(job_id, checkpoint_type=checkpoint_type)
    result = dict(client.poll_request(job_id, request_id))
    parsed = checkpoint_id_from_stage_path(result.get("stage_path"))
    result["checkpoint_id"] = parsed
    return result


def save_recipe_checkpoints(client: CortexTrainingClient, job_id: str) -> dict[str, dict[str, Any]]:
    """Save a weights-only checkpoint (HF tree, plus LoRA adapters under default/)."""
    weights = save_checkpoint(client, job_id, checkpoint_type="weights-only")
    return {"weights-only": weights}


def source_checkpoint_info(
    source_job_id: str | None,
    checkpoint_id: str | None = None,
) -> dict[str, str] | None:
    """Build create-time ``source_checkpoint_info`` from a training job."""
    if checkpoint_id and not source_job_id:
        raise ValueError("checkpoint_id requires source_job_id")
    if source_job_id is None:
        return None
    if not checkpoint_id:
        raise ValueError("source_job_id requires checkpoint_id")
    return {
        "checkpoint_id": str(checkpoint_id).strip(),
        "source_job_id": source_job_id,
    }


def sampling_job_body(
    *,
    model_name: str,
    max_seq_len: int,
    n_gpus: int,
    dtype: str = "bfloat16",
    seed: int = 42,
    gpu_memory_utilization: float = 0.8,
    lora_rank: int = 0,
    source_checkpoint_info: dict[str, str] | None = None,
    debug_image_tag: str | None = None,
) -> dict[str, Any]:
    """Standalone sampling from original HF weights or a weights-only checkpoint."""
    inference_config: dict[str, Any] = {
        "max_seq_len": max_seq_len,
        "n_gpus": n_gpus,
        "vllm_config": {
            "max_model_len": max_seq_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        },
    }

    sub_job: dict[str, Any] = {
        "job_type": "sampling",
        "model_name": model_name,
        "dtype": dtype,
        "seed": seed,
        "inference_config": inference_config,
    }
    if source_checkpoint_info is not None:
        sub_job["source_checkpoint_info"] = dict(source_checkpoint_info)

    body: dict[str, Any] = {"sub_job_configs": [sub_job]}
    if debug_image_tag:
        body["debug"] = {"job": {"image_tag": debug_image_tag}}
    return body


def inference_job_body(
    *,
    model_name: str,
    max_seq_len: int,
    n_gpus: int,
    dtype: str = "bfloat16",
    seed: int = 42,
    gpu_memory_utilization: float = 0.8,
    lora_rank: int = 0,
    source_checkpoint_info: dict[str, str] | None = None,
    training_gpus: int | None = None,
    debug_image_tag: str | None = None,
) -> dict[str, Any]:
    """Sampling from original weights or a weights-only checkpoint."""
    return sampling_job_body(
        model_name=model_name,
        max_seq_len=max_seq_len,
        n_gpus=n_gpus,
        dtype=dtype,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        lora_rank=lora_rank,
        source_checkpoint_info=source_checkpoint_info,
        debug_image_tag=debug_image_tag,
    )


def prepare_inference_weights(
    client: CortexTrainingClient,
    job_id: str,
    job_body: dict[str, Any],
    *,
    lora_rank: int,
) -> dict[str, Any] | None:
    """After a colocated LoRA job starts, sync adapters into sampling."""
    sub_types = {str(sub.get("job_type") or "") for sub in job_body.get("sub_job_configs") or []}
    if "training" not in sub_types or "sampling" not in sub_types:
        return None
    if lora_rank <= 0:
        return None
    logger.info("Syncing LoRA adapters from training into sampling on %s", job_id)
    return sync_weights(client, job_id, weight_format="lora")


def _sampling_cli(
    *,
    module: str,
    config_path: str,
    job_id: str,
    extra: str = "",
) -> str:
    line = f"  python -m {module} config={config_path} source_job_id={job_id}"
    if extra:
        line += extra
    return line


def log_saved_checkpoints(
    *,
    config_path: str,
    job_id: str,
    saved: Mapping[str, Mapping[str, Any]],
    sampling_command: str | None = None,
    lora_rank: int = 0,
    sample_prompt: str | None = None,
    enable_thinking: bool = False,
    renderer_name: str | None = None,
    model_name: str | None = None,
    n_gpus: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    max_examples: int | None = None,
) -> None:
    weights = saved["weights-only"]
    checkpoint_id = str(weights.get("checkpoint_id") or "").strip()
    logger.info(
        "Saved weights-only checkpoint_id=%s tag=%s from job %s.",
        checkpoint_id or None,
        weights.get("checkpoint_tag"),
        job_id,
    )
    if sampling_command == "evaluate":
        module = "recipes.inference.sampling.evaluate"
        verb = "evaluate"
    elif sampling_command == "sample":
        module = "recipes.inference.sampling.sample"
        verb = "sample"
    else:
        return

    extra = ""
    if model_name:
        extra += f" model_name={model_name}"
    if n_gpus is not None:
        extra += f" n_gpus={n_gpus}"
    if checkpoint_id:
        extra += f" checkpoint_id={checkpoint_id}"
    if sampling_command == "evaluate":
        if temperature is not None:
            extra += f" temperature={temperature}"
        if max_tokens is not None:
            extra += f" max_tokens={max_tokens}"
        if top_p is not None:
            extra += f" top_p={top_p}"
        if seed is not None:
            extra += f" seed={seed}"
        if max_examples is not None:
            extra += f" max_examples={max_examples}"
    elif sample_prompt:
        extra += f" prompt={json.dumps(sample_prompt)}"
        if temperature is not None:
            extra += f" temperature={temperature}"
    if enable_thinking:
        extra += " enable_thinking=true"
    if renderer_name:
        extra += f" renderer_name={renderer_name}"

    logger.info(
        "%s this weights-only checkpoint with:\n%s",
        verb,
        _sampling_cli(
            module=module,
            config_path=config_path,
            job_id=job_id,
            extra=extra,
        ),
    )


def recommended_renderer_name(
    model_name: str,
    *,
    enable_thinking: bool | None = None,
    renderer_name: str | None = None,
) -> str:
    """Pick a tinker renderer for ``model_name``.

    ``renderer_name`` wins when set. Otherwise tinker recommended names are
    filtered by ``enable_thinking``:

    - ``False``: ``*_disable_thinking`` when the model has one (Qwen3-8B →
      ``qwen3_disable_thinking``).
    - ``True``: thinking-on renderer (Qwen3-8B → ``qwen3``).
    - ``None``: tinker's top recommendation (thinking on for hybrid Qwen3).
    """
    if renderer_name:
        return renderer_name
    from tinker_cookbook import model_info

    try:
        recommended = list(model_info.get_recommended_renderer_names(model_name))
    except KeyError as exc:
        raise ValueError(
            f"tinker_cookbook has no recommended renderer for {model_name!r}; "
            "see https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/#available-renderers"
        ) from exc
    if not recommended:
        raise ValueError(f"tinker_cookbook listed no renderers for {model_name!r}")
    if enable_thinking is False:
        disabled = [name for name in recommended if name.endswith("_disable_thinking")]
        return disabled[0]
    if enable_thinking is True:
        enabled = [name for name in recommended if not name.endswith("_disable_thinking")]
        return enabled[0]
    return recommended[0]


def build_renderer(
    model_name: str,
    renderer_name: str | None = None,
    *,
    enable_thinking: bool | None = None,
):
    """Return ``(tokenizer, renderer, renderer_name)`` for ``model_name``.

    Uses :func:`recommended_renderer_name` unless ``renderer_name`` is given.
    These recipe helpers only cover models tinker lists; Cortex Training itself can
    host more.
    """
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    resolved = recommended_renderer_name(
        model_name,
        enable_thinking=enable_thinking,
        renderer_name=renderer_name,
    )
    return tokenizer, renderers.get_renderer(resolved, tokenizer), resolved


def stop_params_for(stop_sequences: Sequence[Any]) -> dict:
    token_ids = [int(stop) for stop in stop_sequences if isinstance(stop, int) and not isinstance(stop, bool)]
    strings = [stop for stop in stop_sequences if isinstance(stop, str)]

    params: dict = {}
    if len(token_ids) > 0:
        params["stop_token_ids"] = token_ids
    if len(strings) > 0:
        params["stop"] = strings
    return params


def router_replay_config(
    enabled: bool = True,
    max_cache_bytes: int | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "max_cache_bytes": int(_ROUTER_REPLAY_DEFAULT_MAX_CACHE_BYTES if max_cache_bytes is None else max_cache_bytes),
    }


def router_replay_stop_params(
    stop_sequences: Sequence[Any],
    tokenizer: Any | None = None,
) -> dict:
    params = stop_params_for(stop_sequences)
    strings = [stop for stop in stop_sequences if isinstance(stop, str)]
    if len(strings) == 0:
        return params
    if tokenizer is None:
        raise ValueError(
            "router replay with string stop sequences needs a tokenizer to build dss_stop_token_sequences"
        )
    params["dss_stop_token_sequences"] = [
        [int(token) for token in tokenizer.encode(stop, add_special_tokens=False)] for stop in strings
    ]
    return params


def sampling_params_with_sample_ids(
    base_params: dict,
    sample_ids: Sequence[str],
) -> list[dict]:
    """Per-prompt sampling params carrying ``dss_sample_id`` for router replay."""
    return [{**base_params, "dss_sample_id": sample_id} for sample_id in sample_ids]


def bootstrap_router_replay(
    client: CortexTrainingClient,
    job_id: str,
    max_cache_bytes: int | None = None,
) -> dict:
    training = training_sub_job_id(job_id)
    sampling = sampling_sub_job_id(job_id)
    result = client.bootstrap_router_replay(
        job_id,
        source_sub_job_id=sampling,
        target_sub_job_id=training,
        max_cache_bytes=max_cache_bytes,
        sub_job_id=training,
        sub_job_type="training",
    )
    request_id = result.get("request_id") if isinstance(result, dict) else None
    if request_id is not None:
        return client.poll_request(job_id, request_id)
    return result


def discard_router_replay(
    client: CortexTrainingClient,
    job_id: str,
    sample_ids: Sequence[str],
) -> dict:
    return client.router_replay_discard(
        job_id,
        list(sample_ids),
        sub_job_id=sampling_sub_job_id(job_id),
        sub_job_type="sampling",
    )


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


def use_next_token_labels(model_provider: str) -> bool:
    """Whether SFT labels should already be next-token targets.

    HuggingFace and Liger CausalLM loss shift ``labels`` internally. Cortex Training
    SP SFT does the same shift in ``prepare_sft_request_labels``. Those
    providers need labels aligned with ``input_ids``.

    prime_rl fused CE compares ``logits[i]`` to ``labels[i]`` and does not
    shift, so it needs next-token labels.
    """
    return model_provider == "prime_rl"


def sequence_from_conversation(
    messages: Sequence[Any],
    renderer: Any,
    train_on_what: Any,
    max_seq_len: int | None = None,
    next_token_labels: bool = False,
) -> TrainSequence:
    """Render a chat conversation straight into Cortex Training's forward-backward shape.

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
    if next_token_labels:
        for position in range(len(token_ids) - 1):
            if token_weights[position + 1] > 0.0:
                labels[position] = token_ids[position + 1]
    else:
        for position, (token_id, weight) in enumerate(zip(token_ids, token_weights)):
            if weight > 0.0:
                labels[position] = token_id
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
    temperature: float | None = None,
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
    if temperature is not None:
        kwargs["temperature"] = torch.full(
            (len(sequences), width),
            float(temperature),
            dtype=torch.float32,
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


def _forward_backward_payload(
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    processing: dict | None = None,
    rr_sample_ids: Sequence[str] | None = None,
    rr_discard: bool = True,
    router_replay_sampling_job_id: str | None = None,
) -> bytes:
    frame: dict[str, Any] = {"args": (), "kwargs": kwargs}
    if context:
        frame["context"] = context
    if processing:
        frame["processing"] = processing
    if rr_sample_ids is not None:
        frame["rr_sample_ids"] = list(rr_sample_ids)
        frame["rr_discard"] = bool(rr_discard)

    metadata: dict[str, Any] = {
        "response_options": {"format": "dssst1", "delivery": "chunked"},
    }
    if router_replay_sampling_job_id is not None:
        metadata["router_replay"] = {
            "sampling_job_id": str(router_replay_sampling_job_id),
        }
    elif rr_sample_ids is not None:
        raise ValueError("rr_sample_ids requires router_replay_sampling_job_id (the sampling sub-job id)")
    return wire.dumps(frame, metadata=metadata)


def forward_backward(
    client: CortexTrainingClient,
    job_id: str,
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    processing: dict | None = None,
    rr_sample_ids: Sequence[str] | None = None,
    rr_discard: bool = True,
    router_replay_sampling_job_id: str | None = None,
) -> dict:
    payload = _forward_backward_payload(
        kwargs,
        context=context,
        processing=processing,
        rr_sample_ids=rr_sample_ids,
        rr_discard=rr_discard,
        router_replay_sampling_job_id=router_replay_sampling_job_id,
    )
    request_id = client.forward_backward(job_id, payload)
    return client.poll_request(job_id, request_id)


def forward_loss(
    client: CortexTrainingClient,
    job_id: str,
    kwargs: dict[str, torch.Tensor],
) -> dict:
    return forward_backward(client, job_id, kwargs)


def optimizer_step(
    client: CortexTrainingClient,
    job_id: str,
    learning_rate: float | None = None,
) -> dict:
    request_id = client.step(job_id, learning_rate=learning_rate)
    return client.poll_request(job_id, request_id)


def forward_backward_step(
    client: CortexTrainingClient,
    job_id: str,
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    learning_rate: float | None = None,
    processing: dict | None = None,
    rr_sample_ids: Sequence[str] | None = None,
    rr_discard: bool = True,
    router_replay_sampling_job_id: str | None = None,
) -> tuple[dict, dict]:
    fwd_bwd_result = forward_backward(
        client,
        job_id,
        kwargs,
        context=context,
        processing=processing,
        rr_sample_ids=rr_sample_ids,
        rr_discard=rr_discard,
        router_replay_sampling_job_id=router_replay_sampling_job_id,
    )
    step_result = optimizer_step(client, job_id, learning_rate=learning_rate)
    return fwd_bwd_result, step_result


def sync_weights(
    client: CortexTrainingClient,
    job_id: str,
    weight_format: str | None = None,
) -> dict:
    request_id = client.weight_sync(
        job_id,
        source_sub_job_id=training_sub_job_id(job_id),
        target_sub_job_ids=[sampling_sub_job_id(job_id)],
        weight_format=weight_format,
    )
    return client.poll_request(job_id, request_id)


def _runtime_error_from_logs(client: CortexTrainingClient, job_id: str) -> str | None:
    """Best-effort last ``RuntimeError`` line from a failed job's execution logs."""
    try:
        logs = client.fetch_execution_logs(job_id)
    except Exception:
        logger.warning(
            "Could not fetch logs for %s. Inspect with: cortex-training download-log %s",
            job_id,
            job_id,
        )
        return None
    errors: list[str] = []
    for log in logs:
        content = log.get("content") if isinstance(log, Mapping) else None
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("level") or "").upper() != "ERROR":
                continue
            errors.append(str(payload.get("msg") or line))
    if not errors:
        return None
    matches: list[str] = []
    for text in errors:
        matches.extend(re.findall(r"RuntimeError: .+", text))
    if matches:
        # Wrapper lines like "failed on N node(s)" hide the S3/tag cause.
        unique: list[str] = []
        for match in matches:
            if match not in unique:
                unique.append(match)
        return "\n".join(unique[-5:])
    return errors[-1][:2000]


@contextlib.contextmanager
def running_job(
    client: CortexTrainingClient,
    job_body: dict,
    job_id: str | None = None,
    keep_job: bool | None = None,
) -> Iterator[str]:
    """Yield the id of a running job, releasing its GPUs on the way out.

    Pass ``job_id`` to attach to a job that already exists instead of creating
    one. ``keep_job`` defaults to "keep what I attached to, cancel what I
    created", so a loop pointed at someone else's job never tears it down; set
    it explicitly to override either way.
    """
    attached = job_id is not None
    if keep_job is None:
        keep_job = attached
    if attached:
        logger.info("attaching to job %s; waiting for workers", job_id)
    else:
        job_id = client.create_job_from_body(job_body)["job_id"]
        logger.info("created job %s; waiting for workers", job_id)
    assert job_id is not None
    try:
        client.wait_for_job(job_id)
    except RuntimeError as exc:
        detail = _runtime_error_from_logs(client, job_id)
        if detail:
            raise RuntimeError(f"{exc}\n{detail}") from exc
        raise
    logger.info("job %s is running", job_id)
    try:
        yield job_id
    finally:
        if keep_job:
            logger.info("leaving job %s running", job_id)
        else:
            try:
                client.cancel_job(job_id)
                logger.info("cancelled job %s", job_id)
            except Exception:
                logger.exception("failed to cancel job %s -- check it by hand", job_id)
