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
"""Losses for tabular in-context-learning models (the ``tabular`` loader).

A request carries one or more tables of the same shape: model kwargs ``x`` ``[..., R, C]`` (context rows first),
``y`` ``[..., R_context]`` and ``categorical_mask`` ``[C]``, plus ``context["y_query"]`` ``[..., R_query]`` with
``R_query = R - R_context``. The model returns ``model_outputs["outputs"]`` ``[..., R_query, K]``.

- ``tabular_cross_entropy``: mean cross-entropy of the query rows over the first ``config["num_classes"]`` logits.
  ``y`` and ``y_query`` are class indices in ``[0, num_classes)``.
- ``tabular_pinball``: mean pinball loss of the query rows over the ``K`` quantiles at levels evenly spaced in
  ``[0.001, 0.999]``. ``y`` and ``y_query`` are floats in the model's (preprocessed) target space.

Both are means over the shard's query rows. Data-parallel shards hold equally many tables and DeepSpeed averages
gradients over them, so the model gradient is that of the request-wide mean.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from arctic_platform.rl.processors.base_loss import BaseLoss

OUTPUTS_KEY = "outputs"
Y_QUERY_KEY = "y_query"
ACCURACY_METRIC = "tabular_accuracy"
CORRECT_COUNT_METRIC = "tabular_correct_count"
QUERY_ROW_COUNT_METRIC = "tabular_query_row_count"


def _config(request: dict) -> dict:
    processing = request.get("processing")
    config = processing.get("config") if isinstance(processing, dict) else None
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("tabular losses take processing.config as an object")
    return config


def _query_targets(request: dict, name: str, *, integer: bool) -> torch.Tensor:
    """Validate ``context["y_query"]`` against the request's ``x`` and ``y`` and return it."""
    x, y = request.get("x"), request.get("y")
    if not torch.is_tensor(x) or x.dim() not in (2, 3):
        raise ValueError(f"{name} requires model kwarg x as a [R, C] or [tables, R, C] tensor")
    if not torch.is_tensor(y) or tuple(y.shape[:-1]) != tuple(x.shape[:-2]) or y.shape[-1] >= x.shape[-2]:
        raise ValueError(
            f"{name} requires model kwarg y with x's leading dims and fewer rows than x (the context rows), got "
            f"x={tuple(x.shape)} y={tuple(y.shape) if torch.is_tensor(y) else type(y).__name__}"
        )
    context = request.get("context") or {}
    y_query = context.get(Y_QUERY_KEY)
    expected = (*x.shape[:-2], x.shape[-2] - y.shape[-1])
    if not torch.is_tensor(y_query) or tuple(y_query.shape) != expected:
        received = tuple(y_query.shape) if torch.is_tensor(y_query) else type(y_query).__name__
        raise ValueError(
            f"{name} requires context[{Y_QUERY_KEY!r}] of shape {expected} (the query rows), got {received}"
        )
    for label, tensor in (("y", y), (Y_QUERY_KEY, y_query)):
        if integer and (tensor.is_floating_point() or tensor.is_complex() or tensor.dtype == torch.bool):
            raise ValueError(f"{name} takes {label} as class indices (an integer tensor), got {tensor.dtype}")
        if not integer and not tensor.is_floating_point():
            raise ValueError(f"{name} takes {label} as float targets, got {tensor.dtype}")
        if not integer and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} {label} must be finite")
    return y_query


def _outputs(model_outputs: dict, name: str) -> torch.Tensor:
    outputs = model_outputs.get(OUTPUTS_KEY)
    if not torch.is_tensor(outputs):
        raise ValueError(f"{name} requires model_outputs[{OUTPUTS_KEY!r}], the tabular model's query-row outputs")
    return outputs


class TabularCrossEntropyLoss(BaseLoss):
    name = "tabular_cross_entropy"

    @staticmethod
    def _num_classes(config: dict) -> int:
        num_classes = config.get("num_classes")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(f"tabular_cross_entropy requires processing.config.num_classes >= 2, got {num_classes!r}")
        unknown = set(config) - {"num_classes"}
        if unknown:
            raise ValueError(f"Unknown config keys for loss_fn 'tabular_cross_entropy': {sorted(unknown)}")
        return num_classes

    def batching_callback(self, request: dict) -> None:
        num_classes = self._num_classes(_config(request))
        y_query = _query_targets(request, self.name, integer=True)
        for label, tensor in (("y", request["y"]), (Y_QUERY_KEY, y_query)):
            if tensor.numel() and (tensor.min().item() < 0 or tensor.max().item() >= num_classes):
                raise ValueError(f"tabular_cross_entropy {label} must hold class indices in [0, {num_classes})")

    def loss(
        self,
        model_outputs: dict,
        batch: dict,
        meta: dict,
        config: dict,
        device: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del device
        num_classes = self._num_classes(config)
        outputs = _outputs(model_outputs, self.name)
        if outputs.shape[-1] < num_classes:
            raise ValueError(
                f"tabular_cross_entropy num_classes={num_classes} exceeds the model's {outputs.shape[-1]} outputs"
            )
        logits = outputs[..., :num_classes].float().reshape(-1, num_classes)
        targets = {**meta, **batch}[Y_QUERY_KEY].to(logits.device).long().reshape(-1)
        loss = F.cross_entropy(logits, targets)
        correct = float((logits.detach().argmax(dim=-1) == targets).sum().item())
        metrics = {
            ACCURACY_METRIC: correct / targets.numel(),
            CORRECT_COUNT_METRIC: correct,
            QUERY_ROW_COUNT_METRIC: float(targets.numel()),
        }
        return loss, metrics

    def metrics_callback(self, worker_metrics, metrics: dict) -> None:
        # The counts are summed over data-parallel workers; the accuracy is rebuilt from them.
        if metrics.get(QUERY_ROW_COUNT_METRIC):
            metrics[ACCURACY_METRIC] = metrics[CORRECT_COUNT_METRIC] / metrics[QUERY_ROW_COUNT_METRIC]


class TabularPinballLoss(BaseLoss):
    name = "tabular_pinball"

    def batching_callback(self, request: dict) -> None:
        unknown = set(_config(request))
        if unknown:
            raise ValueError(f"Unknown config keys for loss_fn 'tabular_pinball': {sorted(unknown)}; it takes none")
        _query_targets(request, self.name, integer=False)

    def loss(
        self,
        model_outputs: dict,
        batch: dict,
        meta: dict,
        config: dict,
        device: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del config, device
        quantiles = _outputs(model_outputs, self.name).float()
        targets = {**meta, **batch}[Y_QUERY_KEY].to(quantiles.device).float().unsqueeze(-1)
        levels = torch.linspace(0.001, 0.999, quantiles.shape[-1], device=quantiles.device)
        diff = targets - quantiles
        loss = torch.maximum(levels * diff, (levels - 1) * diff).mean()
        return loss, {QUERY_ROW_COUNT_METRIC: float(targets.numel())}
