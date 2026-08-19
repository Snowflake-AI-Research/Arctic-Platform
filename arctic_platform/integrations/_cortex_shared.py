# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex ``forward-backward`` payload reshape, shared by the SkyRL shim and
the verl adapter (both construct ``{batch, meta, processing}``; Cortex takes
``{args, kwargs, context, processing}``)."""

from __future__ import annotations

from typing import Any


def to_cortex_fwd_bwd_payload(
    batch: dict,
    *,
    dp_size: int,
    processing: dict | None = None,
) -> dict:
    """Reshape ``{batch, meta, processing}`` -> Cortex ``{args, kwargs, context, processing}``.

    Omits ``old_log_probs_shifted``; the server-side GRPO loss restores
    ``π_old = π_new`` via ``logprobs.detach()``. Correct for single-epoch
    on-policy GRPO only — recipes with ``ppo_epochs > 1`` or KL-to-reference
    need the real rollout-time snapshot and should fail before reaching here.
    """
    import torch

    b = batch["batch"] if isinstance(batch, dict) and "batch" in batch else batch
    meta = batch.get("meta", {}) if isinstance(batch, dict) else {}
    processing = processing or (batch.get("processing") if isinstance(batch, dict) else None) or {}

    input_ids = b["input_ids"]
    attention_mask = b.get("attention_mask")
    prompt_len = int(meta.get("prompt_len", 0))
    if not torch.is_tensor(input_ids):
        raise TypeError(f"cortex fwd_bwd: input_ids must be a tensor, got {type(input_ids).__name__}")

    total = int(input_ids.shape[-1])
    response_len = max(total - prompt_len, 1)

    args: list[Any] = [input_ids]
    kwargs: dict[str, Any] = {
        "response_length": response_len,
        "prompt_length": prompt_len,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    context: dict[str, Any] = {
        "advantages": b.get("advantages"),
        "response_mask": b.get("response_mask"),
        "dp_size": int(dp_size),
    }
    context = {k: v for k, v in context.items() if v is not None}

    return {"args": args, "kwargs": kwargs, "context": context, "processing": processing}
