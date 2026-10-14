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

"""Orchestrator micro batches -> one Cortex forward-backward payload.

The orchestrator packs several samples into each micro batch, aligned per token at
the token's own position. Cortex packs on its side and wants one sample per row,
left-aligned (padding at the tail), with the loss tensors in its shifted alignment:
``tensor[:, i]`` scores ``input_ids[:, i + 1]``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from prime_rl.transports.batch import MicroBatch


@dataclass
class _Sample:
    input_ids: np.ndarray
    loss_mask: np.ndarray
    advantages: np.ndarray
    inference_logprobs: np.ndarray


def _unsupported(micro_batch: MicroBatch) -> str | None:
    if micro_batch.mm_refs is not None or micro_batch.mm_token_type_ids is not None:
        return "multimodal samples"
    if micro_batch.sampling_mask is not None:
        return "sampling-mask replay (top-p/top-k truncated sampling)"
    if micro_batch.ce_weights is not None and any(micro_batch.ce_weights):
        return "the ce loss component"
    if micro_batch.ref_kl_weights is not None and any(micro_batch.ref_kl_weights):
        return "the ref_kl loss component"
    return None


def _split_samples(micro_batch: MicroBatch) -> list[_Sample]:
    if (feature := _unsupported(micro_batch)) is not None:
        raise ValueError(f"Cortex training does not support {feature}.")
    loss_mask = np.asarray(micro_batch.loss_mask, dtype=bool)
    if micro_batch.rl_weights is not None:
        rl_weights = np.asarray(micro_batch.rl_weights, dtype=np.float32)
        if not np.isin(rl_weights[loss_mask], (0.0, 1.0)).all():
            raise ValueError("Cortex's grpo loss takes a binary loss mask: rl weights must be 0 or 1.")
        loss_mask &= rl_weights != 0
    temperatures = np.asarray(micro_batch.temperatures, dtype=np.float32)
    if not (temperatures[loss_mask] == 1.0).all():
        raise ValueError("Cortex scores the training tokens at temperature 1: sample them at temperature 1.0.")

    input_ids = np.asarray(micro_batch.input_ids, dtype=np.int64)
    advantages = np.asarray(micro_batch.advantages, dtype=np.float32)
    inference_logprobs = np.asarray(micro_batch.inference_logprobs, dtype=np.float32)
    bounds = np.cumsum([0, *micro_batch.sequence_lengths])
    return [
        _Sample(input_ids[start:end], loss_mask[start:end], advantages[start:end], inference_logprobs[start:end])
        for start, end in zip(bounds[:-1], bounds[1:])
    ]


def build_fwd_bwd_payload(
    micro_batches: list[MicroBatch], *, adv_tau: float, min_rows: int, include_temperature: bool = True
) -> tuple[dict[str, Any], dict[str, float]]:
    """The forward-backward payload of one training step, and batch statistics.

    A sample whose only trainable token is its first predicts nothing and is dropped.
    Cortex hands every data-parallel rank at least one row, so short batches are
    padded to ``min_rows`` with one-token rows that carry no loss.

    ``include_temperature`` adds the per-token temperature tensor the MoE provider's chunked
    LM head requires; the default (dense) provider's forward takes no such kwarg, so pass False.
    """
    samples = [
        sample
        for micro_batch in micro_batches
        for sample in _split_samples(micro_batch)
        if sample.loss_mask[1:].any()
    ]
    if not samples:
        raise ValueError("The training batch has no trainable tokens.")

    num_rows = max(len(samples), min_rows)
    width = max(len(sample.input_ids) for sample in samples)
    input_ids = np.zeros((num_rows, width), dtype=np.int64)
    attention_mask = np.zeros((num_rows, width), dtype=np.int64)
    loss_mask = np.zeros((num_rows, width), dtype=bool)
    advantages = np.zeros((num_rows, width), dtype=np.float32)
    old_log_probs = np.zeros((num_rows, width), dtype=np.float32)
    for row, sample in enumerate(samples):
        length = len(sample.input_ids)
        input_ids[row, :length] = sample.input_ids
        attention_mask[row, :length] = 1
        loss_mask[row, : length - 1] = sample.loss_mask[1:]
        advantages[row, : length - 1] = adv_tau * sample.advantages[1:]
        old_log_probs[row, : length - 1] = sample.inference_logprobs[1:]
    attention_mask[len(samples) :, 0] = 1

    input_ids_tensor = torch.from_numpy(input_ids)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids_tensor,
        "attention_mask": torch.from_numpy(attention_mask),
    }
    if include_temperature:
        # The MoE chunked LM head scales every position's logits by its temperature; all loss
        # tokens were sampled at temperature 1 (see _split_samples).
        kwargs["temperature"] = torch.ones((num_rows, width), dtype=torch.float32)
    payload = {
        "args": (),
        "kwargs": kwargs,
        "context": {
            "input_ids": input_ids_tensor,
            "loss_mask": torch.from_numpy(loss_mask),
            "advantages": torch.from_numpy(advantages),
            "old_log_probs_shifted": torch.from_numpy(old_log_probs),
        },
    }
    stats = {
        "batch/num_samples": float(len(samples)),
        "batch/num_tokens": float(attention_mask[: len(samples)].sum()),
        "batch/num_loss_tokens": float(loss_mask.sum()),
        "batch/max_seq_len": float(width),
        "batch/padded_tokens": float(num_rows * width),
    }
    return payload, stats
