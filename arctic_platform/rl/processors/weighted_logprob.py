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

import torch

from .pipeline import register_loss_fn


@register_loss_fn("weighted_logprob_sum")
def weighted_logprob_sum(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Unnormalized ``sum(w * logprobs)``. Callers pre-scale ``w`` (grad accum, etc.)."""
    del meta, config, device
    logprobs = model_outputs["logprobs"]
    weights = batch["logprob_weights_shifted"]

    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        weights = weights * loss_mask

    loss = (logprobs * weights).sum()
    return loss, {"loss": loss.detach()}
