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

"""Generic ``sum(w * logprobs)`` surrogate for client-side losses."""

from typing import Any
from typing import Sequence

import torch

from .packed_reduction import additive_packed_loss_reduction
from .pipeline import register_loss_fn


def _batch_or_meta_tensor(batch: dict, meta: dict, key: str, *, required: bool = True):
    value = batch.get(key) if key in batch else None
    if value is None:
        value = meta.get(key) if key in meta else None
    if value is None and not required:
        return None
    if not torch.is_tensor(value):
        raise ValueError(f"weighted_logprob_sum requires tensor {key}")
    return value


def _weighted_logprob_sum_packed_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
):
    """Sum per-window ``sum(w * logprobs)`` across packed microbatches.

    Each window is already the unnormalized sum over its own tokens, so the
    scales stay 1 and the reported loss adds. The bare metric name ``loss``
    stays a mean under the platform reducer; ``avg_loss`` is the sum.
    """
    del config, loss_fn_name
    if not microbatches:
        raise ValueError("weighted_logprob_sum packed reduction requires at least one microbatch")
    for index, microbatch in enumerate(microbatches):
        if not torch.is_tensor(microbatch.get("logprob_weights_shifted")):
            raise ValueError(f"weighted_logprob_sum packed microbatch {index} requires tensor logprob_weights_shifted")
    return additive_packed_loss_reduction((1.0,) * len(microbatches))


@register_loss_fn(
    "weighted_logprob_sum",
    packed_loss_reduction=_weighted_logprob_sum_packed_reduction,
)
def weighted_logprob_sum(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Unnormalized ``sum(w * logprobs)``. Callers pre-scale ``w`` (grad accum, etc.)."""
    del config, device
    # Packing keeps model inputs on ``batch`` and moves the other tensors onto
    # ``meta`` for the inner forward. Accept the weights from either side.
    logprobs = model_outputs.get("logprobs")
    if not torch.is_tensor(logprobs):
        raise ValueError("weighted_logprob_sum requires tensor model_outputs['logprobs']")
    weights = _batch_or_meta_tensor(batch, meta, "logprob_weights_shifted")
    loss_mask = _batch_or_meta_tensor(batch, meta, "loss_mask", required=False)
    if loss_mask is not None:
        weights = weights * loss_mask

    loss = (logprobs * weights).sum()
    return loss, {"loss": loss.detach()}
