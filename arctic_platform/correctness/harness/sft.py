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

"""Shared dataset locations, configured SFT settings, and answer-token scoring."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List
from typing import Sequence
from typing import Tuple

from . import gsm8k

TRAIN_SPLIT = Path("/code/shared/gsm8k/train.parquet")
VALIDATION_SPLIT = Path("/code/shared/gsm8k/test.parquet")


def configured_learning_rate(cfg) -> float:
    """The rate the config's own optimizer block declares.

    Not the check's to choose. Ten steps at another rate reach a different set of weights and therefore a
    different validation loss than the operator's job would reach, and the point of the comparison is that
    both engines read the weights this config produces.
    """
    optimizer = cfg.training.get("optimizer")
    rate = optimizer.get("lr") if isinstance(optimizer, dict) else None
    if rate is None:
        raise ValueError(
            f"{cfg.config_id}: the training config declares no optimizer learning rate. This check trains "
            "the config as written, so there is nothing to run rather than a rate to pick."
        )
    return float(rate)


def configured_rows_per_step(cfg) -> int:
    """The global batch size the config trains at, which is how many dataset rows a step consumes."""
    rows = cfg.training.get("train_batch_size")
    if rows is None:
        raise ValueError(f"{cfg.config_id}: the training config declares no train_batch_size")
    return int(rows)


def cross_entropy(per_row_answer_logprobs: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """Negative mean log-probability over every answer token in the batch.

    The one reduction both sides use. Each side is responsible only for producing, per row, that row's
    answer-token log-probabilities in order; the sign, the denominator and the pooling across rows are
    decided here, once.
    """
    total = 0.0
    count = 0
    for row in per_row_answer_logprobs:
        for value in row:
            total += float(value)
            count += 1
    if count == 0:
        raise ValueError("no answer tokens were scored; a cross entropy over nothing is not a measurement")
    return -total / count, count


def reference_answer_logprobs(logprobs, examples: Sequence[gsm8k.Example]) -> List[List[float]]:
    """The training zone's log-probabilities over each row's answer span.

    The route returns one entry per position, scoring the label at that position. The labels it was sent
    are logit aligned, so the entry at ``s`` is the log-probability of the token at ``s + 1``. A row of
    length ``L`` whose answer begins at ``p`` therefore has its answer scored by entries ``p - 1`` through
    ``L - 2``.
    """
    if logprobs.ndim != 2 or logprobs.shape[0] != len(examples):
        raise RuntimeError("forward-only logprobs must have one two-dimensional row per validation example")
    rows: List[List[float]] = []
    for index, example in enumerate(examples):
        _assert_answer_span(example, index)
        span = [float(value) for value in logprobs[index, example.n_prompt - 1 : example.length - 1].tolist()]
        if len(span) != example.answer_tokens:
            raise RuntimeError(
                f"row {index}: the forward-only response covers {len(span)} of the row's "
                f"{example.answer_tokens} answer tokens"
            )
        if not all(math.isfinite(value) for value in span):
            raise RuntimeError(f"row {index}: forward-only answer log-probabilities must be finite")
        rows.append(span)
    return rows


def _assert_answer_span(example: gsm8k.Example, index: int) -> None:
    if not 1 <= example.n_prompt < example.length:
        raise RuntimeError(f"row {index}: answer span must follow a prompt token and contain at least one token")
