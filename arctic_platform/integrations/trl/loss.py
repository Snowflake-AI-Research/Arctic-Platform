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

"""The single loss function the TRL integration needs.

TRL computes its own loss on the client and sends back ``d(loss)/d(logprobs)``. The server backpropagates
``sum(w * logprobs)``, a first-order surrogate whose gradient with respect to every parameter equals the
gradient of the real loss. So the loss itself never crosses the wire.

This is what lets one entry cover every TRL loss variant. Compare ``integrations/verl/grpo_loss.py``, which is
523 lines reimplementing verl's ``masked_mean`` / ``agg_loss`` / ``compute_policy_loss_vanilla`` on this side
and still supports only ``loss_mode="vanilla"`` out of the twelve entries in verl's ``POLICY_LOSS_REGISTRY``.
Nothing here needs revisiting when TRL adds a loss.

Equivalent framing: a weighted cross-entropy ``sum(-logprobs * weights)`` is the same surrogate under
``weights = -d(loss)/d(logprobs)``. Tinker takes that route and maps its custom-loss path onto plain
``cross_entropy`` without adding a loss at all.
"""

from typing import Any

import torch

from arctic_platform.rl.processors.pipeline import register_loss_fn


@register_loss_fn("weighted_logprob_sum")
def weighted_logprob_sum(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``sum(w * logprobs)`` over the loss mask.

    ``logprob_weights_shifted`` follows the same roll(-1) convention as every other log-prob tensor in the
    batch, so it lines up with ``model_outputs["logprobs"]`` without a shift here.

    The weights arrive already scaled for gradient accumulation: TRL divides by its accumulation step count
    before the gradient reaches the client. The sum below must therefore stay unnormalized, or the two
    scalings compose.
    """
    logprobs = model_outputs["logprobs"]
    weights = batch["logprob_weights_shifted"]

    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        weights = weights * loss_mask

    loss = (logprobs * weights).sum()
    return loss, {"loss": loss.detach()}
