# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""VeRL → Cortex wire reshape. Keep ``old_log_probs_shifted``; loss_fn=grpo."""

from __future__ import annotations

import os
from typing import Any

import torch


def _drop_old_log_probs() -> bool:
    """Omit ``old_log_probs_shifted`` so the zone defaults π_old ≡ π_new.

    ``grpo_loss`` falls back to ``logprobs.detach()`` when the key is absent,
    which pins the importance ratio at exactly 1 -- the same on-policy shape TRL
    and SkyRL send. Kept as an A/B control that isolates the IS term.
    """
    return os.environ.get("CORTEX_VERL_DROP_OLD_LOGPROBS", "0") not in ("0", "", "false", "False")


def _to_predict_next(t):
    """Response-aligned ``[B, S]`` -> the zone's predict-next layout.

    ``compute_logprobs`` scores ``labels = roll(input_ids, -1)``, so slot ``i``
    of the zone's ``logprobs`` holds ``log P(input_ids[i + 1])``. verl keeps
    ``old_log_probs`` / ``advantages`` / ``response_mask`` aligned to the token
    they belong to, so each per-token tensor moves one slot left before the zone
    can compare it against those logprobs. Without this the ratio becomes
    ``exp(log P(tok i+1) - log P(tok i))`` and the importance weight explodes.

    Same contract as verl's ``shift_nested_response_aligned_to_predict_next``.
    A slice copy rather than ``torch.roll`` so no value wraps across the row.
    """
    if not torch.is_tensor(t) or t.dim() < 1 or t.shape[-1] < 2:
        return t
    out = torch.zeros_like(t)
    out[..., :-1] = t[..., 1:]
    return out


def _left_align(tensors: dict, attention_mask, extra: dict):
    mask = attention_mask.to(torch.bool)
    width = mask.shape[-1]
    lengths = mask.sum(dim=1)
    valid = torch.arange(width, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    if torch.equal(mask, valid):
        return tensors, extra, attention_mask
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)

    def move(t):
        if not torch.is_tensor(t) or t.dim() != 2 or t.shape[-1] != width:
            return t
        pad = False if t.dtype == torch.bool else 0
        return torch.where(valid, t.gather(1, order), torch.full_like(t, pad))

    return {k: move(v) for k, v in tensors.items()}, {k: move(v) for k, v in extra.items()}, valid.to(attention_mask.dtype)


def _restore_from_left_align(aligned, attention_mask):
    """Scatter left-aligned ``[B, T]`` logprobs back onto the caller's pad layout."""
    if not torch.is_tensor(aligned):
        aligned = torch.as_tensor(aligned)
    mask = attention_mask.to(torch.bool)
    width = mask.shape[-1]
    if aligned.dim() != 2 or aligned.shape[-1] != width:
        return aligned
    lengths = mask.sum(dim=1)
    valid = torch.arange(width, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    if torch.equal(mask, valid):
        return aligned
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
    restored = torch.zeros_like(aligned)
    restored.scatter_(1, order, aligned)
    return torch.where(mask, restored, torch.zeros_like(restored))


def to_cortex_fwd_bwd_payload(batch: dict, *, processing: dict | None = None) -> dict:
    payload = dict(batch)
    processing_in = processing or payload.pop("processing", None)
    if "batch" in payload and isinstance(payload["batch"], dict):
        tensors = dict(payload["batch"])
        meta = dict(payload.get("meta") or {})
    else:
        tensors = dict(payload)
        meta = {}

    input_ids = tensors.get("input_ids")
    attention_mask = tensors.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("cortex fwd_bwd requires input_ids and attention_mask")
    loss_mask = tensors.pop("loss_mask", None)
    if loss_mask is None:
        loss_mask = tensors.pop("response_mask", None)
    if loss_mask is None:
        raise ValueError("cortex fwd_bwd requires loss_mask or response_mask")
    if torch.is_tensor(loss_mask):
        loss_mask = loss_mask.to(torch.bool)
    advantages = tensors.pop("advantages", None)
    if advantages is None:
        raise ValueError("cortex fwd_bwd requires advantages")
    old_log_probs = tensors.pop("old_log_probs", None)

    forwarded = {"input_ids": input_ids}
    if "position_ids" in tensors:
        forwarded["position_ids"] = tensors["position_ids"]
    extra = {"advantages": advantages, "loss_mask": loss_mask}
    if old_log_probs is not None:
        extra["old_log_probs"] = old_log_probs
    forwarded, scored, attention_mask = _left_align(forwarded, attention_mask, extra)
    input_ids = forwarded["input_ids"]
    context: dict[str, Any] = {
        "input_ids": input_ids,
        "advantages": _to_predict_next(scored["advantages"]),
        "loss_mask": _to_predict_next(scored["loss_mask"]),
    }
    if "old_log_probs" in scored and not _drop_old_log_probs():
        context["old_log_probs_shifted"] = _to_predict_next(scored["old_log_probs"])
    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "position_ids" in forwarded:
        kwargs_out["position_ids"] = forwarded["position_ids"]
    caller = dict((processing_in or {}).get("config") or {})
    proc_config = {"eps_clip": 0.2, "loss_agg_mode": "token-mean", "entropy_coeff": 0.0, **caller}
    for k in ("global_batch_size", "batch_num_tokens"):
        if k not in proc_config and k in meta:
            proc_config[k] = int(meta[k])
    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": context,
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }


def actor_policy_metrics(old_lp, new_lp, loss_mask) -> dict[str, float]:
    if not torch.is_tensor(old_lp):
        old_lp = torch.as_tensor(old_lp)
    if not torch.is_tensor(new_lp):
        new_lp = torch.as_tensor(new_lp)
    if not torch.is_tensor(loss_mask):
        loss_mask = torch.as_tensor(loss_mask)
    width = min(old_lp.shape[-1], new_lp.shape[-1], loss_mask.shape[-1])
    mask = loss_mask[..., :width].to(dtype=torch.bool)
    old_lp = old_lp[..., :width]
    new_lp = new_lp[..., :width]
    if not bool(mask.any()):
        raise ValueError("empty loss_mask")
    ratio = torch.exp(new_lp - old_lp)
    return {
        "actor/ppo_kl": float((old_lp - new_lp)[mask].mean().item()),
        "actor/ratio_mean": float(ratio[mask].mean().item()),
        "actor/max_abs_ratio_minus_1": float((ratio[mask] - 1).abs().max().item()),
    }
