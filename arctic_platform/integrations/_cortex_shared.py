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
"""Reshape SkyRL's ``{batch, meta, processing}`` into Cortex's
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
    the live forward. A reference-model request has no such escape, so
    ``_CortexClientShim.fwd_no_grad`` refuses it rather than zero-filling a KL
    term into a function of π_new alone.

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


# Padding sentinel per tensor. `labels` uses HF's ignore_index so a padded
# column contributes no loss even if something downstream reads labels directly.
_PAD_VALUE: dict[str, Any] = {"labels": -100}


def _left_align_batch(tensors: dict, attention_mask, extra: dict) -> tuple[dict, dict, Any]:
    """Move every row's real tokens into its leading columns, padding at the tail.

    Cortex's data plane packs microbatches and rejects any other layout:
    ``pack_microbatch`` raises "packing requires left-aligned rows: every row's
    real tokens must be its leading columns, with padding at the tail". SkyRL
    hands us the opposite, left-padding each row to the longest sequence in the
    batch, so its payload is never safe to forward as-is.

    The rewrite is driven entirely by ``attention_mask``: a stable sort on
    validity yields, per row, the real columns in their original order followed
    by the pad columns. Every full-width tensor is gathered through that one
    index, so ``advantages`` and ``loss_mask`` keep pointing at the tokens they
    scored. Re-aligning ``input_ids`` alone would displace the loss by a
    different amount on every row -- no error, no crash, just a gradient
    computed against the wrong tokens.

    Returns the rewritten tensors, the rewritten extras, and the new mask.
    """
    import torch

    mask = attention_mask.to(torch.bool)
    width = mask.shape[-1]
    lengths = mask.sum(dim=1)
    valid = torch.arange(width, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)

    # Already conformant (a batch whose rows all happen to be full width, or a
    # caller that aligned upstream): skip the gather rather than pay for it.
    if torch.equal(mask, valid):
        return tensors, extra, attention_mask

    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)

    def move(name: str, t):
        if not torch.is_tensor(t) or t.dim() != 2 or t.shape[-1] != width:
            return t
        pad = _PAD_VALUE.get(name, False if t.dtype == torch.bool else 0)
        return torch.where(valid, t.gather(1, order), torch.full_like(t, pad))

    return (
        {k: move(k, v) for k, v in tensors.items()},
        {k: move(k, v) for k, v in extra.items()},
        valid.to(attention_mask.dtype),
    )


def to_cortex_fwd_bwd_payload(batch: dict, *, processing: dict | None = None) -> dict:
    """Reshape a SkyRL fwd_bwd payload into Cortex's wire shape.

    ``old_log_probs_shifted`` is dropped: server-side GRPO defaults it to
    ``logprobs.detach()`` (π_old ≡ π_new), correct for single-epoch on-policy.

    Requires ``loss_mask`` or ``response_mask``; falling back to
    ``attention_mask`` would train on prompt tokens.

    Rows are left-aligned on the way out (see :func:`_left_align_batch`) --
    Cortex's packer rejects any other layout.
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

    forwarded = {"input_ids": input_ids}
    for k in ("position_ids", "labels"):
        if k in tensors:
            forwarded[k] = tensors[k]
    forwarded, scored, attention_mask = _left_align_batch(
        forwarded, attention_mask, {"advantages": advantages, "loss_mask": loss_mask}
    )
    input_ids = forwarded["input_ids"]
    advantages, loss_mask = scored["advantages"], scored["loss_mask"]

    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for k in ("position_ids", "labels"):
        if k in forwarded:
            kwargs_out[k] = forwarded[k]

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
