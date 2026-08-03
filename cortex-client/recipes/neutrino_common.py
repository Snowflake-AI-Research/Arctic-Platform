from __future__ import annotations

import contextlib
import io
import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dss_client import NeutrinoClient, wire

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100


def make_client(config_path: str, **overrides: Any) -> NeutrinoClient:
    parsed = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    config = parsed.get("connection", parsed)

    pat = config.get("pat")
    kwargs: dict[str, Any] = dict(
        database=config.get("database", "NEUTRINO_DB"),
        schema=config.get("schema", "PUBLIC"),
        endpoint=config.get("endpoint", "cortex-training"),
        poll_interval=float(config.get("poll_interval", 0.5)),
        poll_timeout=float(config.get("poll_timeout", 1800.0)),
    )
    kwargs.update(overrides)

    host = config.get("host")
    if host is None:
        raise ValueError(
            "connection config needs `host` (Snowflake PAT auth)"
        )
    if pat is None:
        raise ValueError(
            "no PAT found: put `pat` in the connection config"
        )
    return NeutrinoClient.from_pat(
        host=host,
        pat=pat,
        verify_ssl=bool(config.get("verify_ssl", True)),
        **kwargs,
    )


def training_sub_job_id(job_id: str) -> str:
    return f"{job_id}:training:0"


def sampling_sub_job_id(job_id: str) -> str:
    return f"{job_id}:sampling:0"


def build_renderer(model_name: str, renderer_name: str | None = None):
    """Return ``(tokenizer, renderer, renderer_name)`` for ``model_name``.

    ``tinker_cookbook.model_info`` only knows the models tinker itself serves,
    so an unlisted model needs an explicit ``renderer_name``.
    """
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    if renderer_name is None:
        try:
            renderer_name = model_info.get_recommended_renderer_name(model_name)
        except KeyError as exc:
            raise ValueError(
                f"tinker_cookbook has no recommended renderer for {model_name!r}; "
                "pass renderer_name= explicitly (e.g. renderer_name=qwen3)"
            ) from exc
    return tokenizer, renderers.get_renderer(renderer_name, tokenizer), renderer_name


def stop_params_for(stop_sequences: Sequence[Any]) -> dict:
    token_ids = [
        int(stop)
        for stop in stop_sequences
        if isinstance(stop, int) and not isinstance(stop, bool)
    ]
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
                f"input_ids ({len(self.input_ids)}) and labels ({len(self.labels)}) "
                "must have the same length"
            )

def sequence_from_conversation(
    messages: Sequence[Any],
    renderer: Any,
    train_on_what: Any,
    max_seq_len: int | None = None,
) -> TrainSequence:
    """Render a chat conversation straight into Neutrino's forward-backward shape.

    ``renderer.build_supervised_example`` tokenizes the whole conversation and
    returns per-token weights aligned with those tokens: ``weights[i] > 0`` marks
    token ``i`` as one the model should learn to produce. It covers every
    assistant turn in one sequence, which is the reason for using a renderer at
    all -- ``apply_chat_template(return_assistant_tokens_mask=True)`` only works
    for templates carrying ``{% generation %}`` markers, and Qwen3's does not
    (HF then returns an all-zero mask).
    """
    model_input, weights = renderer.build_supervised_example(
        list(messages), train_on_what=train_on_what
    )
    token_ids = [int(token) for token in model_input.to_ints()]
    token_weights = [float(weight) for weight in weights.tolist()]
    if len(token_ids) != len(token_weights):
        raise ValueError(
            f"renderer returned {len(token_ids)} tokens but {len(token_weights)} weights"
        )

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
        position_ids=torch.arange(width, dtype=torch.long)
        .expand(len(sequences), -1)
        .contiguous(),
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


def forward_backward_step(
    client: NeutrinoClient,
    job_id: str,
    kwargs: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor] | None = None,
    learning_rate: float | None = None,
    processing: dict | None = None,
) -> tuple[dict, dict]:
    frame: dict[str, Any] = {"args": (), "kwargs": kwargs}
    if context:
        frame["context"] = context
    if processing:
        frame["processing"] = processing
    payload = wire.dumps(
        frame,
        metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
    )
    request_id = client.forward_backward(job_id, payload)
    fwd_bwd_result = client.poll_request(job_id, request_id)
    request_id = client.step(job_id, learning_rate=learning_rate)
    step_result = client.poll_request(job_id, request_id)
    return fwd_bwd_result, step_result


def sync_weights(client: NeutrinoClient, job_id: str) -> dict:
    """Push the training sub-job's weights into the sampling sub-job."""
    request_id = client.weight_sync(
        job_id,
        source_sub_job_id=training_sub_job_id(job_id),
        target_sub_job_ids=[sampling_sub_job_id(job_id)],
    )
    return client.poll_request(job_id, request_id)


@contextlib.contextmanager
def running_job(
    client: NeutrinoClient,
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
    client.wait_for_job(job_id)
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
