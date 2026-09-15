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

"""Cortex ``compute_logprobs`` post: chunked log-softmax + gather.

Registered as ``cortex_compute_logprobs``. AP's tiled / FlashAttn + entropy
path stays ``compute_entropy_and_logprobs`` / ``ap_compute_logprobs``.
"""

from __future__ import annotations

import math
import os

import torch

from arctic_platform.common.registry import register_post_processor

# Target fp32 slice size for the chunked log_softmax. Peak memory of the fp32
# intermediate is chunk_tokens * V * 4 bytes; deriving the shard count from this
# budget keeps the slice size constant as V varies across models.
_LOGPROB_SLICE_BYTES = 1 * (2**30)  # 1 GiB


def _eager_log_softmax_gather(logits_chunk: torch.Tensor, labels_chunk: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32, log_softmax over vocab, gather the label column."""
    lp = torch.log_softmax(logits_chunk.float(), dim=-1)
    return lp.gather(-1, labels_chunk.unsqueeze(-1)).squeeze(-1)


def _log_softmax_gather(logits_chunk: torch.Tensor, labels_chunk: torch.Tensor) -> torch.Tensor:
    """Eager on CPU; Cortex ``torch.compile`` fuse when CUDA is available."""
    impl = _eager_log_softmax_gather
    if os.environ.get("DSS_LOGPROB_COMPILE", "1") != "0" and torch.cuda.is_available():
        impl = torch.compile(_eager_log_softmax_gather)
        globals()["_log_softmax_gather"] = impl
    return impl(logits_chunk, labels_chunk)


@register_post_processor("cortex_compute_logprobs")
def compute_logprobs_post(model_outputs: dict, batch: dict, meta: dict, device: str) -> dict:
    """Compute per-token log-probs from logits using the Cortex kernel.

    Peak memory is dominated by the fp32 log_softmax intermediate. Shard the
    token axis so each slice fits ``_LOGPROB_SLICE_BYTES``. Prefer request
    ``labels`` (SP-aligned) over ``roll(input_ids)``. Zero ignore_index=-100
    positions. Pass through when the model already returned ``logprobs``.
    """
    del device
    if "logprobs" in model_outputs:
        return {}

    logits = model_outputs.get("logits")
    if logits is None:
        raise ValueError(
            "cortex_compute_logprobs requires model outputs containing either 'logprobs' or 'logits'"
        )

    context = {**meta, **batch}
    labels = context.get("labels")
    if labels is None:
        input_ids = context.get("input_ids")
        if input_ids is None:
            return {}
        input_ids = input_ids.to(logits.device)
        if input_ids.ndim < logits.ndim:
            input_ids = input_ids.view(logits.shape[:-1])
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        labels = labels.to(logits.device)
        if labels.ndim < logits.ndim:
            if labels.numel() != math.prod(logits.shape[:-1]):
                raise ValueError(
                    "context labels are incompatible with logits shape: "
                    f"labels={tuple(labels.shape)} logits={tuple(logits.shape)}"
                )
            labels = labels.view(logits.shape[:-1])

    valid_labels = labels != -100
    safe_labels = labels.masked_fill(~valid_labels, 0)

    logits_2d = logits.reshape(-1, logits.shape[-1])
    labels_1d = safe_labels.reshape(-1)
    n_tokens, vocab_size = logits_2d.shape
    if n_tokens == 0:
        return {"logprobs": torch.empty_like(labels, dtype=logits.dtype)}

    bytes_per_token_fp32 = vocab_size * 4
    num_shards = math.ceil(n_tokens * bytes_per_token_fp32 / _LOGPROB_SLICE_BYTES)
    num_shards = max(1, min(num_shards, n_tokens))

    logit_shards = torch.chunk(logits_2d, num_shards, dim=0)
    label_shards = torch.chunk(labels_1d, num_shards, dim=0)
    out = [_log_softmax_gather(ls, ys) for ls, ys in zip(logit_shards, label_shards)]

    logprobs = torch.cat(out).view_as(labels)
    logprobs = torch.where(valid_labels, logprobs, torch.zeros_like(logprobs))
    return {"logprobs": logprobs}
