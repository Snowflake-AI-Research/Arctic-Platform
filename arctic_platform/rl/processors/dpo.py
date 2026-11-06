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

"""Sequence-sum sigmoid DPO with pair-preserving execution."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.distributed as dist
import torch.nn.functional as F

from arctic_platform.common.utils.batch import flatten_batch_rows
from arctic_platform.common.utils.batch import group_batch_rows
from arctic_platform.common.utils.batch import reconstruct_position_ids_
from arctic_platform.common.utils.batch import unpack_batch

from .base_loss import REQUIRES_ALIGNED_TOKEN_LOGPROBS
from .base_loss import USES_TOKEN_BUDGET_PIPELINE
from .base_loss import BaseLoss
from .functional import _per_sequence_loss_totals
from .microbatch import DEFAULT_MAX_TOKENS_PER_MB
from .packed_reduction import additive_packed_loss_reduction


class DPOLoss(BaseLoss):
    """Adjacent chosen/rejected rows, frozen reference sums, one global pair mean."""

    name = "dpo"
    capabilities = frozenset({REQUIRES_ALIGNED_TOKEN_LOGPROBS, USES_TOKEN_BUDGET_PIPELINE})

    def __init__(self):
        self.global_pairs = None

    @staticmethod
    def _beta(config):
        beta = config.get("beta", 0.1)
        if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(beta) or beta <= 0:
            raise ValueError("dpo beta must be finite and positive")
        beta_fp32 = torch.tensor(beta, dtype=torch.float32)
        if not torch.isfinite(beta_fp32) or beta_fp32 <= 0:
            raise ValueError("dpo beta must be representable as a positive finite FP32 value")
        return beta

    def validation_callback(self, context, config):
        context = flatten_batch_rows(context)
        self._beta(config)
        pair_ids = context.get("dpo_pair_ids")
        chosen = context.get("dpo_is_chosen")
        reference = context.get("dpo_reference_logps")
        if not torch.is_tensor(pair_ids) or pair_ids.dtype != torch.int64 or pair_ids.ndim != 1:
            raise ValueError("dpo_pair_ids must be an int64 row vector")
        rows = pair_ids.numel()
        if rows == 0 or rows % 2:
            raise ValueError("dpo requires a nonempty even number of rows")
        if not torch.equal(pair_ids[0::2], pair_ids[1::2]) or pair_ids[0::2].unique().numel() != rows // 2:
            raise ValueError("dpo_pair_ids must identify unique adjacent pairs")
        if not torch.is_tensor(chosen) or chosen.dtype != torch.bool or chosen.shape != pair_ids.shape:
            raise ValueError("dpo_is_chosen must be a bool row vector")
        if not chosen[0::2].all() or chosen[1::2].any():
            raise ValueError("dpo rows must alternate chosen then rejected")
        if (
            not torch.is_tensor(reference)
            or not reference.is_floating_point()
            or reference.shape != pair_ids.shape
            or not torch.isfinite(reference).all()
        ):
            raise ValueError("dpo_reference_logps must be finite floating-point sequence sums")
        mask = context.get("loss_mask")
        labels = context.get("labels")
        if not torch.is_tensor(mask) or mask.dtype != torch.bool:
            raise ValueError("dpo loss_mask must be a bool response prediction mask")
        if (
            not torch.is_tensor(labels)
            or labels.dtype != torch.int64
            or labels.numel() != mask.numel()
            or (context.get("cu_seqlens") is None and labels.shape != mask.shape)
        ):
            raise ValueError("dpo labels must be int64 and aligned with loss_mask")
        labels = labels.reshape(mask.shape)
        if ((labels < 0) & (labels != -100)).any() or ((labels == -100) & mask).any():
            raise ValueError("dpo response prediction positions require valid labels")
        cu = context.get("cu_seqlens")
        if cu is None:
            if mask.ndim != 2 or mask.shape[0] != rows:
                raise ValueError("dpo loss_mask must have shape [rows, tokens]")
        elif (
            not torch.is_tensor(cu)
            or cu.dtype not in (torch.int32, torch.int64)
            or cu.ndim != 1
            or cu.numel() != rows + 1
            or cu[0] != 0
            or (cu[1:] <= cu[:-1]).any()
            or cu[-1] != mask.numel()
        ):
            raise ValueError("dpo requires complete sequence boundaries for every packed row")

    def batching_callback(self, request):
        hosted = "batch" not in request and "kwargs" not in request
        payload = request
        if hosted:
            payload = {
                "kwargs": {key: value for key, value in request.items() if key not in {"context", "processing"}},
                "context": request.get("context", {}),
                "processing": request.get("processing", {}),
            }
        _, batch, meta, processing = unpack_batch(payload)
        config = processing["config"] = processing.get("config") or {}
        if not isinstance(batch, dict):
            raise ValueError("dpo requires one complete global pair batch, not a GAS list")
        packing_keys = ("cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k", "seq_idx", "max_length_q", "max_length_k")
        if any(key in batch or key in meta for key in packing_keys):
            raise ValueError("dpo global requests must be padded rows without caller cu_seqlens or varlen boundaries")
        if set(config) - {"beta"}:
            raise ValueError("dpo request config accepts only beta")
        self.validation_callback({**meta, **batch}, config)
        if meta.get("load_balancer") or meta.get("zorro_train_enable"):
            raise ValueError("dpo requires pair-preserving packing without row rebalancing or ZoRRO")
        reference = batch["dpo_reference_logps"].detach().double()
        reference_margin = (reference[0::2] - reference[1::2]).float()
        beta = self._beta(config)
        if not torch.isfinite(reference_margin * beta).all():
            raise ValueError("dpo reference margins must remain finite in FP32 loss arithmetic")
        ids = batch.get("input_ids")
        attn = batch.get("attention_mask")
        if (
            not torch.is_tensor(ids)
            or ids.dtype != torch.int64
            or ids.ndim != 2
            or ids.shape != batch["loss_mask"].shape
        ):
            raise ValueError("dpo input_ids must be int64 [rows, tokens] with the same shape as loss_mask")
        if (
            not torch.is_tensor(attn)
            or attn.shape != ids.shape
            or attn.dtype not in (torch.bool, torch.int32, torch.int64)
            or not ((attn == 0) | (attn == 1)).all()
        ):
            raise ValueError("dpo attention_mask must be a binary integer/bool mask")
        if (ids < 0).any():
            raise ValueError("dpo input_ids must be non-negative token IDs")
        lengths = attn.sum(-1)
        if (lengths == 0).any() or (attn[:, 1:] > attn[:, :-1]).any():
            raise ValueError("dpo requires nonempty right-padded token rows")
        for positions in (batch.get("position_ids"), meta.get("position_ids")):
            if positions is None:
                continue
            valid = attn.bool()
            expected = attn.long().cumsum(-1) - 1
            if (
                not torch.is_tensor(positions)
                or positions.dtype != torch.int64
                or positions.shape != ids.shape
                or not torch.equal(positions[valid], expected[valid])
            ):
                raise ValueError("dpo position_ids must start at zero and increase by one within each valid row")
        if (batch["loss_mask"] & ~attn.bool()).any():
            raise ValueError("dpo response mask cannot select padding")
        budget = meta.get("max_tokens_per_mb", DEFAULT_MAX_TOKENS_PER_MB)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("max_tokens_per_mb must be a positive integer")
        if (lengths.reshape(-1, 2).sum(-1) > budget).any():
            raise ValueError("a dpo pair exceeds max_tokens_per_mb; increase the budget")
        self.global_pairs = ids.shape[0] // 2
        batch["dpo_reference_logps"] = batch["dpo_reference_logps"].detach()
        if "position_ids" in meta:
            batch["position_ids"] = meta.pop("position_ids")
        if meta.get("drop_position_ids", False) and "position_ids" not in batch:
            reconstruct_position_ids_(batch)
        batch = group_batch_rows(batch, 2)
        request.clear()
        if hosted:
            request.update(batch)
            request.update(context=meta, processing=processing)
        else:
            request.update(batch=batch, meta=meta, processing=processing)

    def model_forward_callback(self, model_kwargs, context, config, output_keys):
        # These targets are logit-aligned for score extraction, not the model's
        # internally shifted supervised objective.
        if not model_kwargs.get("dss_compute_logprobs", False):
            model_kwargs.pop("labels", None)

    @staticmethod
    def _require_finite(finite):
        invalid = (~finite).to(dtype=torch.int32)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        if invalid.item():
            raise ValueError("dpo response logprobs and loss arithmetic must remain finite on every rank")

    def backward_validation_callback(self, loss):
        self._require_finite(torch.isfinite(loss).all())

    def packed_reduction_callback(self, microbatches, config, loss_fn_name):
        return additive_packed_loss_reduction([mb["dpo_pair_ids"].numel() // 2 for mb in microbatches])

    def loss(self, model_outputs, batch, meta, config, device):
        del device
        context = flatten_batch_rows({**meta, **batch})
        if self.global_pairs is None:
            raise ValueError("dpo requires whole-request batching before execution")
        scores = model_outputs.get("logprobs")
        if not torch.is_tensor(scores) or not scores.is_floating_point():
            raise ValueError("dpo requires floating-point token logprobs")
        mask = context["loss_mask"].to(scores.device)
        if scores.numel() != mask.numel():
            raise ValueError("dpo logprobs must align with response prediction positions")
        score_dtype = scores.dtype
        scores = scores.reshape(mask.shape)
        scores_finite = torch.isfinite(scores[mask]).all()
        scores = scores if scores.dtype == torch.float64 else scores.float()
        sums, _ = _per_sequence_loss_totals(scores, mask, context.get("cu_seqlens"))
        # Subtract frozen sums before narrowing: large equal FP64 sums cancel
        # exactly rather than independently overflowing to FP32 infinities.
        refs = context["dpo_reference_logps"].detach().to(device=scores.device, dtype=torch.float64)
        reference_margin = (refs[0::2] - refs[1::2]).to(scores.dtype)
        margins = (sums[0::2] - sums[1::2]) - reference_margin
        beta = self._beta(config)
        scaled_margins = beta * margins
        losses = -F.logsigmoid(scaled_margins)
        contribution = losses.sum() / self.global_pairs
        dp_size = context.get("dp_size", 1)
        if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size <= 0:
            raise ValueError("dpo dp_size must be a positive integer")
        scalar = contribution * dp_size
        # All ranks call this once per synchronized packed window. Reject
        # overflow together before any rank enters backward collectives.
        derivative_bound = scores.new_tensor(beta) * (dp_size / self.global_pairs)
        finite = (
            scores_finite
            & torch.isfinite(scaled_margins).all()
            & torch.isfinite(scalar)
            & torch.isfinite(derivative_bound.to(score_dtype)).all()
        )
        self._require_finite(finite)
        return scalar, {"dpo_loss_sum": contribution.detach().item(), "dpo_pair_count": float(losses.numel())}

    def output_callback(self, model_outputs):
        super().output_callback(model_outputs)
        model_outputs.pop("logprobs", None)
        model_outputs.pop("loss", None)

    def reporting_callback(self, worker_metrics, metrics, avg_loss):
        return sum(worker["dpo_loss_sum"] for worker in worker_metrics)

    def metrics_callback(self, worker_metrics, metrics):
        metrics["dpo_loss"] = sum(worker["dpo_loss_sum"] for worker in worker_metrics)
