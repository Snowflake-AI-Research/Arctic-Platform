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
"""Lower a verl-GRPO forward-backward batch onto Cortex's wire shape.

`fwd_bwd_request` notes that the call signature is unified across backends but
`batch`'s *content* is not: on-prem takes a pre-tokenized verl-GRPO
``{batch, meta}`` while Cortex takes an RPC-style
``{args, kwargs, context}``. This is the lowering for that gap, applied by the
Cortex transport so callers stop branching on backend.

The processing block matches ``recipes/rl/standalone/rl_loop.py`` (and Jae's
original cookbook). ``dp_size`` is deliberately not sent: the server treats it
as a loss divisor, which scales the effective learning rate down by that factor
at multi-GPU data parallelism.
"""

from __future__ import annotations

from typing import Any

# Match the standalone RL recipe and verl's actor.yaml defaults; caller-supplied
# processing.config values win.
_DEFAULT_PROC_CONFIG: dict[str, Any] = {
    "eps_clip": 0.2,
    "loss_agg_mode": "token-mean",
    "entropy_coeff": 0.0,
}


def is_cortex_shaped(batch: dict) -> bool:
    """True when the caller already built Cortex's RPC frame itself.

    The standalone recipes construct ``{args, kwargs, context}`` directly, so
    they must pass through untouched.
    """
    return "kwargs" in batch or "args" in batch


def lower_fwd_bwd_batch(batch: dict, *, processing: dict | None = None) -> dict:
    """Reshape a verl/SkyRL forward-backward payload into Cortex's wire shape.

    ``old_log_probs`` is dropped: server-side GRPO defaults π_old to
    ``logprobs.detach()`` (π_old == π_new), which is correct for single-epoch
    on-policy training and avoids shipping a tensor the server recomputes.

    Requires ``loss_mask`` or ``response_mask``. Falling back to
    ``attention_mask`` would put prompt tokens in the loss and silently corrupt
    the gradient, so it raises instead.
    """
    import torch

    payload = dict(batch)
    processing_in = processing or payload.pop("processing", None)
    payload.pop("router_replay", None)

    if isinstance(payload.get("batch"), dict):
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
            "cortex fwd_bwd requires either 'loss_mask' or 'response_mask' in the batch. "
            "Falling back to 'attention_mask' would include prompt tokens in the loss "
            "and silently corrupt the gradient."
        )
    if torch.is_tensor(loss_mask):
        loss_mask = loss_mask.to(torch.bool)

    advantages = tensors.pop("advantages", None)
    if advantages is None:
        raise ValueError("cortex fwd_bwd requires 'advantages' [B, S]")
    tensors.pop("old_log_probs", None)

    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in ("position_ids", "labels"):
        if key in tensors:
            kwargs_out[key] = tensors[key]

    caller = dict(processing_in or {})
    proc_config = {**_DEFAULT_PROC_CONFIG, **dict(caller.get("config") or {})}
    for key in ("global_batch_size", "batch_num_tokens"):
        if key not in proc_config and key in meta:
            proc_config[key] = int(meta[key])

    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": {"input_ids": input_ids, "advantages": advantages, "loss_mask": loss_mask},
        "processing": {
            # `loss_fn` and `post` are part of the lowering, not caller choices:
            # this frame carries advantages/loss_mask in `context`, which is the
            # contract server-side `grpo` reads. verl asks for `verl_grpo`, whose
            # meta contract (actor_config, policy_loss_config) we deliberately do
            # not send, so honouring that request would mis-target the loss.
            # compute_logprobs is what puts per-token logprobs in the response's
            # `batch`; without it the caller gets loss and metrics only.
            "post": ["compute_logprobs"],
            "loss_fn": "grpo",
            "config": proc_config,
        },
    }
