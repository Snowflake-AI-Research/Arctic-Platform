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

"""TRL-shaped server loss. The generic ``weighted_logprob_sum`` surrogate lives in ``rl.processors``."""

from typing import Any

import torch

from arctic_platform.rl.processors.pipeline import register_loss_fn
from arctic_platform.rl.processors.weighted_logprob import weighted_logprob_sum  # noqa: F401


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


_CLIENT_LOSS_ENCODINGS = ("weighted_logprob_sum", "grpo")


def _surrogate_payload(
    encoding: str,
    batch: dict,
    weights: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_fn: str,
) -> tuple[dict, str, dict]:
    """Express ``w = dL/dlogprobs`` as a batch the server can differentiate.

    Both encodings ask for the same thing -- a loss whose gradient wrt log-probs is
    exactly ``w`` -- and differ only in which registered loss carries it.

    ``weighted_logprob_sum`` states it directly, but the server has to have this
    package installed. ``grpo`` gets there with a loss every RL server already
    ships: its gradient wrt log-probs is ``-advantages * ratio``, so pinning
    ``old_log_probs_shifted`` to the log-probs the forward pass just returned makes
    the ratio 1 and leaves ``-advantages``.

    Scale follows from ``token-mean``, which is ``masked_sum / batch_num_tokens *
    dp_size``. Setting both to 1 collapses it to a plain masked sum, so ``grpo``
    reduces to exactly what ``weighted_logprob_sum`` computes -- including how it
    behaves under data parallelism, which is inherited here rather than redefined.

    Clipping cannot engage: at ratio 1, ``min(r*A, clip(r)*A)`` is ``A`` for any
    epsilon. The server recomputes log-probs in bf16, so the ratio lands at 1 only
    to within forward nondeterminism, which perturbs the gradient by ~1e-5
    relative -- far inside the clip band, but not bit-exact.
    """
    if encoding == "grpo":
        return (
            {
                **batch,
                "old_log_probs_shifted": old_log_probs,
                "advantages": -weights,
                "loss_mask": batch["attention_mask"].to(weights.dtype),
            },
            "grpo",
            {"batch_num_tokens": 1, "dp_size": 1},
        )
    return {**batch, "logprob_weights_shifted": weights}, loss_fn, {}
