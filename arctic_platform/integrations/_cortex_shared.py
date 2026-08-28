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
"""Reshape verl/SkyRL ``{batch, meta, processing}`` into Cortex's
``{args, kwargs, context, processing}`` wire format.

The processing block matches ``cortex-client/recipes/rl_loop.py``
(Jae's cookbook). ``dp_size`` is deliberately NOT sent: the server treats it
as a loss divisor, which scales the effective LR down by that factor at
multi-GPU DP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import torch

# Match Jae's rl_loop.py and verl's actor.yaml defaults; caller-supplied
# processing.config values win.
_DEFAULT_PROC_CONFIG: dict[str, Any] = {
    "eps_clip": 0.2,
    "loss_agg_mode": "token-mean",
    "entropy_coeff": 0.0,
}


def zero_logprobs_like(batch: dict) -> torch.Tensor:
    """A ``[B, T]`` float32 zero tensor sized from ``batch``'s ``input_ids``.

    Cortex exposes no ``/forward``, so a caller asked for log-probs has nothing
    to call and substitutes zeros. That is only sound for GRPO without KL, where
    nothing consumes the values: ``to_cortex_fwd_bwd_payload`` drops
    ``old_log_probs`` and the server defaults π_old to ``logprobs.detach()`` from
    the live forward. The verl adapter enforces the preconditions in
    ``_validate_cortex_compat``; the SkyRL path refuses reference-model requests
    in ``_CortexClientShim.fwd_no_grad``.

    Consequence worth knowing before reading a training curve: because π_old is
    re-derived per call rather than snapshotted, the ratio is exactly 1 on every
    minibatch and PPO clipping never engages. A recipe that splits a batch into
    minibatches (SkyRL's ``policy_mini_batch_size`` < ``train_batch_size``) is
    therefore running unclipped updates, not clipped GRPO. It trains, but it is
    not the same objective.

    Callers name their own response keys -- SkyRL's legacy client reads
    ``logprobs``/``entropies`` while verl reads ``log_probs``/``entropy`` -- so
    only the sizing is shared.
    """
    import torch

    body = batch.get("batch") if isinstance(batch, dict) else None
    if not isinstance(body, dict):
        body = batch if isinstance(batch, dict) else {}
    ids = body.get("input_ids")
    b, t = (int(ids.shape[0]), int(ids.shape[-1])) if torch.is_tensor(ids) else (1, 1)
    return torch.zeros((b, max(t, 1)), dtype=torch.float32)


def to_cortex_fwd_bwd_payload(batch: dict, *, processing: dict | None = None) -> dict:
    """Reshape a verl/SkyRL fwd_bwd payload into Cortex's wire shape.

    ``old_log_probs_shifted`` is dropped: server-side GRPO defaults it to
    ``logprobs.detach()`` (π_old ≡ π_new), correct for single-epoch on-policy.

    Requires ``loss_mask`` or ``response_mask``; falling back to
    ``attention_mask`` would train on prompt tokens.
    """
    import torch

    payload = dict(batch)
    processing_in = processing or payload.pop("processing", None)
    payload.pop("router_replay", None)

    if "batch" in payload and isinstance(payload["batch"], dict):
        tensors, meta = dict(payload["batch"]), dict(payload.get("meta") or {})
    else:
        tensors = dict(payload)
        meta = dict(tensors.pop("context", None) or {})

    input_ids = tensors.get("input_ids")
    attention_mask = tensors.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("cortex fwd_bwd requires 'input_ids' and 'attention_mask'")

    loss_mask = tensors.pop("loss_mask", None)
    if loss_mask is None:
        loss_mask = tensors.pop("response_mask", None)
    if loss_mask is None:
        raise ValueError(
            "cortex fwd_bwd requires either 'loss_mask' or 'response_mask' in the "
            "batch. Falling back to 'attention_mask' would include prompt tokens "
            "in the loss and silently corrupt the gradient."
        )
    if torch.is_tensor(loss_mask):
        loss_mask = loss_mask.to(torch.bool)
    advantages = tensors.pop("advantages", None)
    if advantages is None:
        raise ValueError("cortex fwd_bwd requires 'advantages' [B, S]")
    tensors.pop("old_log_probs", None)

    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for k in ("position_ids", "labels"):
        if k in tensors:
            kwargs_out[k] = tensors[k]

    caller_config = dict((processing_in or {}).get("config") or {})
    proc_config: dict[str, Any] = {**_DEFAULT_PROC_CONFIG, **caller_config}
    for k in ("global_batch_size", "batch_num_tokens"):
        if k not in proc_config and k in meta:
            proc_config[k] = int(meta[k])

    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": {"input_ids": input_ids, "advantages": advantages, "loss_mask": loss_mask},
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }
