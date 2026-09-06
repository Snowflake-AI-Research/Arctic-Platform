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

"""Sparse logit gather for TRL async distillation.

``gather_token_ids`` is ``[B, S, K]`` in the same frame as model logits
(next-token / roll(-1) aligned). The surrogate ``weighted_gathered_logit_sum``
is ``sum(w * gathered_logits)`` so TRL can evaluate JSD in-process and ship
``dL/d(logits_k)``.
"""

from __future__ import annotations

import torch

from arctic_platform.common.registry import register_loss_fn
from arctic_platform.common.registry import register_post_processor


@register_post_processor("gather_logits_at_ids")
def gather_logits_at_ids_post(model_outputs: dict, batch: dict, meta: dict, device: str) -> dict:
    """Gather ``logits[..., ids]`` without sending the full vocab over the wire."""
    del meta, device
    if "logits" not in model_outputs:
        raise ValueError("gather_logits_at_ids requires model_outputs['logits']")
    ids = batch.get("gather_token_ids")
    if ids is None:
        raise ValueError("gather_logits_at_ids requires batch['gather_token_ids'] of shape [B, S, K]")
    logits = model_outputs["logits"]
    if not torch.is_tensor(ids):
        ids = torch.as_tensor(ids, device=logits.device)
    else:
        ids = ids.to(device=logits.device)
    if ids.ndim != 3:
        raise ValueError(f"gather_token_ids must be [B, S, K], got {tuple(ids.shape)}")
    if ids.shape[:2] != logits.shape[:2]:
        raise ValueError(
            f"gather_token_ids leading dims {tuple(ids.shape[:2])} != logits {tuple(logits.shape[:2])}"
        )
    gathered = torch.gather(logits, dim=-1, index=ids.long())
    return {"gathered_logits": gathered}


@register_loss_fn("weighted_gathered_logit_sum")
def weighted_gathered_logit_sum(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """First-order surrogate: ``sum(logit_weights * gathered_logits)``."""
    del meta, config, device
    logits_k = model_outputs.get("gathered_logits")
    if logits_k is None:
        raise ValueError("weighted_gathered_logit_sum requires post=['gather_logits_at_ids']")
    weights = batch.get("logit_weights")
    if weights is None:
        raise ValueError("weighted_gathered_logit_sum requires batch['logit_weights']")
    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights, device=logits_k.device, dtype=logits_k.dtype)
    else:
        weights = weights.to(device=logits_k.device, dtype=logits_k.dtype)
    if weights.shape != logits_k.shape:
        raise ValueError(f"logit_weights {tuple(weights.shape)} != gathered_logits {tuple(logits_k.shape)}")
    loss = (logits_k * weights).sum()
    return loss, {"gathered_logit_sum": float(loss.detach())}
