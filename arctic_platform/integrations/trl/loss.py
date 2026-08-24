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

"""``sum(w * logprobs)`` surrogate; TRL's real loss stays on the client."""

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
    """Unnormalized ``sum(w * logprobs)``; TRL already scaled ``w`` for grad accum."""
    logprobs = model_outputs["logprobs"]
    weights = batch["logprob_weights_shifted"]

    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        weights = weights * loss_mask

    loss = (logprobs * weights).sum()
    return loss, {"loss": loss.detach()}


@register_loss_fn("trl_grpo")
def trl_grpo(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Server-side GRPO loss reproducing TRL ``AsyncGRPOTrainer.compute_loss``.

    Pure clipped surrogate (no dual-clip / KL / ref) evaluated in the same
    fused ``fwd_bwd`` as the model forward. The adapter pre-aligns
    ``old_log_probs``/``advantages``/``loss_mask`` into the server's roll(-1)
    logprob frame, so no response-window reshift is done here.

    Normalization mirrors TRL's ``loss / tokens_per_rank / grad_accum`` and adds
    ``* dp_size`` so the gradient is correct after DeepSpeed's cross-DP averaging
    (identical to verl ``agg_loss`` token-mean).
    """
    logprobs = model_outputs["logprobs"]
    old_log_probs = batch["old_log_probs"].to(logprobs.dtype)
    advantages = batch["advantages"].to(logprobs.dtype)
    mask = batch["loss_mask"].to(logprobs.dtype)

    eps_low = float(meta["epsilon_low"])
    eps_high = float(meta["epsilon_high"])

    ratio = torch.exp(logprobs - old_log_probs)
    per_token = -torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantages,
    )

    # Inline masked-sum (keep TRL integration self-contained; do not import verl).
    masked = torch.where(mask.bool(), per_token, torch.zeros_like(per_token)).sum()

    batch_num_tokens = float(meta["batch_num_tokens"])
    dp_size = float(meta.get("dp_size", 1))
    grad_accum = float(meta.get("grad_accum_steps", 1))
    loss = masked / batch_num_tokens * dp_size / grad_accum

    return loss, {"loss": loss.detach()}
