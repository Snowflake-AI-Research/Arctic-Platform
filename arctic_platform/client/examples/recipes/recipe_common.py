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
"""Shared recipe helpers: connection config, chat rendering, batch collation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from arctic_platform.client import CortexConfig

IGNORE_INDEX = -100


def cortex_backend(config_path: str) -> CortexConfig:
    """Build a `CortexConfig` from a connection json (`{"connection": {...}}` or flat)."""
    parsed = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    conn = parsed.get("connection", parsed)
    return CortexConfig(
        host=conn.get("host"),
        base_url=conn.get("base_url"),
        pat=conn.get("pat"),
        database=conn.get("database", "NEUTRINO_DB"),
        schema=conn.get("schema", "PUBLIC"),
        endpoint=conn.get("endpoint", "cortex-training"),
    )


def build_renderer(model_name: str) -> tuple[Any, Any, str]:
    """Return ``(tokenizer, renderer, renderer_name)`` for ``model_name``.

    Uses tinker ``model_info.get_recommended_renderer_name``. These recipe
    helpers only cover models tinker lists; the backend itself can host more.
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


@dataclass
class TrainSequence:
    input_ids: list[int]
    labels: list[int]

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
    """Render a chat conversation straight into the forward-backward shape.

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


def collate(
    sequences: Sequence[TrainSequence],
    pad_token_id: int,
    max_seq_len: int,
    pad_to_max_seq_len: bool = False,
) -> dict[str, Any]:
    """Pad sequences into the RPC-style ``{"args", "kwargs"}`` body Cortex expects."""
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
    for sequence in sequences:
        padding = width - len(sequence.input_ids)
        input_ids.append(sequence.input_ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(sequence.input_ids) + [0] * padding)
        labels.append(sequence.labels + [IGNORE_INDEX] * padding)

    kwargs = dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        position_ids=torch.arange(width, dtype=torch.long).expand(len(sequences), -1).contiguous(),
        labels=torch.tensor(labels, dtype=torch.long),
        use_cache=False,
    )
    return {"args": [], "kwargs": kwargs}
