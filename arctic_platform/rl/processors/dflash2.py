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
"""DFlash2 draft-training objective for the processing path.

The model (``DFlash2OnlineModel``) computes the objective itself, with SpecForge's ``OnlineDFlashModel``, because anchors,
block masks and the target context are all per-sequence. This loss only routes the supervision mask and the per-step
options into the forward, and reports the model's ratio metrics as additive ``loss_term_<name>__num`` /
``loss_term_<name>__den`` pairs. Those are summed over packed windows and data-parallel ranks, so the client divides
them to get exact pooled ratios (the names are SpecForge's metric names, for example
``dflash2/selector/greedy_accepted_length``).

Normalisation: each window's loss is ``(sum w*CE + alpha*sum w*covered*selCE) / sum w`` over that window's blocks.
Windows of a rank are combined weighted by their supervised-token counts, and ranks are averaged by the data-parallel
gradient reduction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .base_loss import BaseLoss
from .packed_reduction import PackedLossReduction
from .packed_reduction import local_mean_packed_loss_reduction

TERMS_OUTPUT_KEY = "dflash2_terms"


class DFlash2Loss(BaseLoss):
    name = "dflash2"

    def model_forward_callback(
        self,
        model_kwargs: dict,
        context: dict,
        config: dict,
        output_keys: list[str],
    ) -> None:
        loss_mask = context.get("loss_mask")
        if not torch.is_tensor(loss_mask):
            raise ValueError("dflash2 requires context['loss_mask'] marking supervised tokens (1) and others (0)")
        # The data plane packs rows into one window; its mask may be [T] while the model takes [1, T].
        model_kwargs["loss_mask"] = loss_mask.reshape(1, -1)
        alpha = config.get("selector_loss_alpha")
        if alpha is not None:
            model_kwargs["selector_loss_alpha"] = float(alpha)
        # Detailed diagnostics (accepted lengths, top-k recall) cost an extra vocabulary projection; they default on.
        model_kwargs["collect_detailed_metrics"] = bool(config.get("detailed_metrics", True))
        model_kwargs["per_position_metrics"] = bool(config.get("per_position_metrics", False))
        output_keys.append(TERMS_OUTPUT_KEY)

    def validation_callback(self, context: dict, config: dict) -> None:
        loss_mask = context.get("loss_mask")
        input_ids = context.get("input_ids")
        if torch.is_tensor(loss_mask) and torch.is_tensor(input_ids) and loss_mask.numel() != input_ids.numel():
            raise ValueError(
                f"dflash2 loss_mask has {loss_mask.numel()} entries but input_ids has {input_ids.numel()}"
            )
        alpha = config.get("selector_loss_alpha")
        if alpha is not None and (not isinstance(alpha, (int, float)) or alpha < 0):
            raise ValueError(f"dflash2 config selector_loss_alpha must be a non-negative number, got {alpha!r}")

    def packed_reduction_callback(
        self,
        microbatches: Sequence[dict],
        config: dict,
        loss_fn_name: str,
    ) -> PackedLossReduction:
        del config, loss_fn_name
        weights = []
        for microbatch in microbatches:
            loss_mask = microbatch.get("loss_mask")
            weights.append(float(loss_mask.sum()) if torch.is_tensor(loss_mask) else 1.0)
        return local_mean_packed_loss_reduction(weights)

    def output_callback(self, model_outputs: dict) -> None:
        model_outputs.pop("logits", None)
        model_outputs.pop("loss", None)  # a scalar has no per-token axis to unpack in a response
        model_outputs.pop(TERMS_OUTPUT_KEY, None)

    def loss(
        self,
        model_outputs: dict,
        batch: dict,
        meta: dict,
        config: dict,
        device: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del batch, meta, config, device
        loss = model_outputs.get("loss")
        terms = model_outputs.get(TERMS_OUTPUT_KEY)
        if loss is None or terms is None:
            raise ValueError("dflash2 needs the DFlash2 online model: its forward returns 'loss' and 'dflash2_terms'")
        metrics: dict[str, Any] = {}
        for name, (numerator, denominator) in terms.items():
            metrics[f"loss_term_{name}__num"] = float(numerator)
            metrics[f"loss_term_{name}__den"] = float(denominator)
        return loss, metrics
