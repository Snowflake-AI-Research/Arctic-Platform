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

"""Arctic Platform backend for TRL's ``TrainingClientProtocol``.

TRL's protocol (``trl/experimental/api/training_client.py``) is two methods::

    forward(model, input_ids, position_ids, completion_mask, aux_loss_coef) -> ForwardOutput
    backward(grad_log_probs) -> None

The trainer computes its loss between the two calls and hands back ``d(loss)/d(log_probs)``. Everything
algorithm-shaped stays on the TRL side, so this file has no notion of GRPO, advantages, or clipping ratios,
and it does not grow when TRL adds a loss variant.

The split into two methods is a split in TRL's interface, not a requirement on the backend: ``backward`` below
issues the fused ``fwd_bwd`` call. Two round trips are inherent regardless, since the weights are a function of
the log probs and cannot be known before the first forward.
"""

from typing import Any

import torch
from trl.experimental.api import ForwardOutput


class ArcticTrainingClient:
    """Runs the forward and the backward on an Arctic RL server while TRL keeps the loss.

    Configuration lives on this object rather than in TRL's config: the trainer takes the client as a
    constructed instance, so endpoint, temperature and loss name never become TRL config fields.

    Args:
        client: A :class:`~arctic_platform.client.client.SyncArcticRLClient`.
        temperature: Sampling temperature applied by the ``apply_temperature`` post-processor.
        loss_fn: Name of the surrogate loss. ``pipeline._resolve_fn`` falls back to a dotted-path import, so
            ``"arctic_platform.integrations.trl.loss.weighted_logprob_sum"`` also resolves when the registry
            entry is not present.
    """

    def __init__(
        self,
        client: Any,
        temperature: float = 1.0,
        loss_fn: str = "weighted_logprob_sum",
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.loss_fn = loss_fn
        self._batch: dict | None = None

    def forward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        aux_loss_coef: float = 0.0,
    ) -> ForwardOutput:
        # `model` is the trainer's local module. Arctic owns the weights, so it is unused here.
        #
        # AsyncGRPOTrainer packs a rank's samples into one padding-free row and marks sequence boundaries by
        # position_ids resets. The server wants padded [B, S] rows, so unpack first. This conversion is the
        # bulk of the adapter and is the main reason it belongs on this side of the wire.
        batch = _unpack_to_padded_rows(input_ids, position_ids, completion_mask)
        self._batch = batch

        response = self.client.fwd_no_grad(
            batch,
            processing={
                "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                "loss_fn": None,
            },
        )
        out = response["batch"]

        # Hand back a leaf tensor: the trainer's `loss.backward()` stops here and deposits
        # d(loss)/d(log_probs) into `.grad`, which is what arrives at `backward` below.
        #
        # The server returns log probs under the roll(-1) convention and TRL expects the [:, 1:] shift, so
        # the repack aligns them rather than merely reshaping.
        return ForwardOutput(
            log_probs=_repack_to_row(out["logprobs"], completion_mask).requires_grad_(True),
            entropy=_repack_to_row(out["entropy"], completion_mask),
            aux_loss=None,  # Arctic reports the MoE aux loss as a metric, not as a differentiable tensor
        )

    def backward(self, grad_log_probs: torch.Tensor) -> None:
        assert self._batch is not None, "backward() called without a preceding forward()"

        batch = dict(self._batch)
        # `_shifted` marks the roll(-1) convention, matching the other log-prob tensors in the batch.
        batch["logprob_weights_shifted"] = _unpack_to_padded(grad_log_probs)

        self.client.fwd_bwd(
            batch,
            processing={
                "post": ["apply_temperature"],
                "loss_fn": self.loss_fn,
            },
        )
        self._batch = None


class ArcticOptimizer(torch.optim.Optimizer):
    """Drives ``client.step()``. Passed to the trainer as ``optimizers=(ArcticOptimizer(...), scheduler)``.

    TRL needs nothing for this. ``transformers.Trainer`` already accepts a user-supplied optimizer and calls
    ``.step()`` on it, so the optimizer leg needs no protocol of its own and gradient clipping stays inside
    ``step()`` where Arctic already does it.
    """

    def __init__(self, client: Any, params: Any) -> None:
        super().__init__(params, {})
        self.client = client
        self.last_grad_norm: float | None = None

    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        metrics = self.client.step().get("metrics", {})
        norm = metrics.get("grad_norm")
        # grad_norm comes back per DP rank as a flat list. Under ZeRO-3 the value is already globally
        # reduced, so the first entry is the value.
        self.last_grad_norm = norm[0] if isinstance(norm, list) else norm

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        pass  # gradients live on the server and are cleared by step()


# ------------------------------------------------------------------------------------------------------- #
# Batch layout helpers
# ------------------------------------------------------------------------------------------------------- #
# Stubs. Padding-free row to padded rows and back is mechanical but fiddly, and is where a first integration
# would actually spend its time. Note what is absent compared with `integrations/verl/adapter.py`: no
# `old_log_probs`, no `advantages`, no `ref_log_prob`. TRL consumed those before the gradient was formed.


def _unpack_to_padded_rows(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> dict:
    raise NotImplementedError


def _unpack_to_padded(tensor: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def _repack_to_row(tensor: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
