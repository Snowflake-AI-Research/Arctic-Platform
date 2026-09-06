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

"""Pad ``RolloutSample``s into a server batch with roll(-1) teacher support."""

from __future__ import annotations

import torch

from arctic_platform.integrations.trl_distill.types import RolloutSample


def max_support(samples: list[RolloutSample]) -> int:
    width = 0
    for sample in samples:
        for ids in sample.teacher_token_ids:
            width = max(width, len(ids))
    return width


def samples_to_train_batch(
    samples: list[RolloutSample],
    pad_token_id: int,
    *,
    max_seq_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad rollouts to ``[B, S]`` and place top-k ids on completion-predicting slots.

    ``gather_token_ids[b, t, :]`` aligns with logits at ``t`` (predicts
    ``input_ids[b, t+1]``). Completion tokens occupy
    ``[prompt_len-1, prompt_len-1+T)``.
    """
    if not samples:
        raise ValueError("samples_to_train_batch got no rollouts")
    rows = []
    for sample in samples:
        full_ids = list(sample.prompt_ids) + list(sample.completion_ids)
        if max_seq_len is not None and len(full_ids) > max_seq_len:
            raise ValueError(f"sequence length {len(full_ids)} exceeds max_seq_len {max_seq_len}")
        if len(sample.teacher_token_ids) != len(sample.completion_ids):
            raise ValueError("teacher_token_ids must align with completion_ids")
        rows.append(full_ids)
    batch_size = len(rows)
    seq_len = max(len(ids) for ids in rows)
    support = max_support(samples)
    if support < 1:
        raise ValueError("teacher support is empty")

    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    gather_token_ids = torch.zeros((batch_size, seq_len, support), dtype=torch.long)
    teacher_logprobs = torch.full((batch_size, seq_len, support), float("-inf"))

    for row_index, (sample, full_ids) in enumerate(zip(samples, rows)):
        length = len(full_ids)
        input_ids[row_index, :length] = torch.tensor(full_ids, dtype=torch.long)
        attention_mask[row_index, :length] = 1
        start = len(sample.prompt_ids) - 1
        stop = start + len(sample.completion_ids)
        loss_mask[row_index, start:stop] = True
        for offset, (ids, lps) in enumerate(zip(sample.teacher_token_ids, sample.teacher_logprobs)):
            width = len(ids)
            gather_token_ids[row_index, start + offset, :width] = torch.tensor(ids, dtype=torch.long)
            teacher_logprobs[row_index, start + offset, :width] = torch.tensor(lps)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "gather_token_ids": gather_token_ids,
        "teacher_logprobs": teacher_logprobs,
    }
