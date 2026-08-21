# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared Cortex-specific payload reshape used by both integration adapters.

The Cortex ``forward-backward`` op consumes an RPC-style
``{args, kwargs, context, processing}`` frame, while both SkyRL and verl
construct verl-GRPO's ``{batch, meta, processing}``. This helper reshapes
between them.

The processing block matches the wire contract in
``arctic_platform/cortex-client/recipes/rl_loop.py::processing_block``
(Jae's cookbook), which is the reference for what Cortex's GRPO loss reads:

    processing = {
        "loss_fn": "grpo",
        "config": {
            "eps_clip": 0.2,
            "loss_agg_mode": "token-mean",
            "entropy_coeff": 0.0,
            "global_batch_size": <actual datum count>,
        },
    }

Deviating from this contract silently changes what the server computes for
the same input. In particular we do NOT send ``dp_size``: on-prem doesn't,
Jae's cookbook doesn't, and the Cortex server treats it as a divisor -- so
sending it at multi-GPU DP scales the effective LR down by that factor.
"""

from __future__ import annotations

from typing import Any

# Defaults match Jae's rl_loop.py and verl's actor.yaml. Callers can
# override any entry by passing them in the incoming ``processing.config``.
_DEFAULT_PROC_CONFIG: dict[str, Any] = {
    "eps_clip": 0.2,
    "loss_agg_mode": "token-mean",
    "entropy_coeff": 0.0,
}


def to_cortex_fwd_bwd_payload(batch: dict, *, processing: dict | None = None) -> dict:
    """Reshape a verl/SkyRL ``{batch, meta, processing}`` payload into Cortex's
    canonical ``{args, kwargs, context, processing}``.

    ``old_log_probs_shifted`` is intentionally omitted -- the server-side GRPO
    loss defaults it to ``logprobs.detach()`` (π_old ≡ π_new), which is
    correct for single-epoch on-policy GRPO.

    ``response_mask`` (or a caller-supplied ``loss_mask``) is required: we do
    NOT fall back to ``attention_mask`` because that includes prompt tokens
    and would train on them (silently wrong gradient).
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

    # Caller wins for any explicit config key; unset keys get the Jae-cookbook
    # defaults. Never inject dp_size / prox_logp_method: neither is in the
    # cookbook and dp_size causes multi-DP LR shrink.
    caller_config = dict((processing_in or {}).get("config") or {})
    proc_config: dict[str, Any] = {**_DEFAULT_PROC_CONFIG, **caller_config}
    # global_batch_size falls back to verl's meta; leave unset if unavailable so
    # the server uses its own count.
    if "global_batch_size" not in proc_config and "global_batch_size" in meta:
        proc_config["global_batch_size"] = int(meta["global_batch_size"])
    if "batch_num_tokens" not in proc_config and "batch_num_tokens" in meta:
        proc_config["batch_num_tokens"] = int(meta["batch_num_tokens"])

    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": {"input_ids": input_ids, "advantages": advantages, "loss_mask": loss_mask},
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }
