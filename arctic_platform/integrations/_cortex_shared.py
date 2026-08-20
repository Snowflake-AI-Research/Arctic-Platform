# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared Cortex-specific payload reshape used by both integration adapters.

Cortex's ``forward-backward`` op takes ``{args, kwargs, context, processing}``
while both SkyRL and verl construct ``{batch, meta, processing}``. This helper
reshapes between them so the SkyRL shim and the verl adapter agree on the
Cortex wire format. Lives under :mod:`arctic_platform.integrations` because
only integration adapters need it — the client / transport layer is unaware.
"""

from __future__ import annotations

from typing import Any


def to_cortex_fwd_bwd_payload(batch: dict, *, dp_size: int, processing: dict | None = None) -> dict:
    """Reshape a verl/SkyRL ``{batch, meta, processing}`` payload into Cortex's
    canonical ``{args, kwargs, context, processing}``. ``old_log_probs_shifted``
    is intentionally omitted -- the server-side GRPO loss defaults it to
    ``logprobs.detach()`` (π_old ≡ π_new), correct for single-epoch on-policy.
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
        loss_mask = attention_mask
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

    proc_config = dict((processing_in or {}).get("config") or {})
    proc_config.setdefault("eps_clip", 0.2)
    proc_config.setdefault("prox_logp_method", "recompute")
    proc_config.setdefault("dp_size", int(dp_size or 1))
    for k in ("batch_num_tokens", "global_batch_size"):
        if k not in proc_config and k in meta:
            proc_config[k] = int(meta[k])

    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": {"input_ids": input_ids, "advantages": advantages, "loss_mask": loss_mask},
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }
