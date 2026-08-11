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

TRL's protocol is a single fused call::

    forward_backward(model, input_ids, position_ids, completion_mask, loss_fn, aux_loss_coef) -> ForwardBackwardOutput

``loss_fn`` is a Python callable mapping per-token log probs to a scalar. It closes over everything
algorithm-shaped (advantages, old log probs, the mask, clipping bounds), so nothing algorithm-shaped reaches
this file and nothing here changes when TRL adds a loss variant.

The adapter runs in the trainer's process even though the model does not, so ``loss_fn`` is called here, on
tensors the server returned. Only tensors cross the wire. That is what lets a TRL user write a new objective
as a plain Python function and run it against Arctic on day one, with no change on this side.

Two wire calls, ``fwd_no_grad`` then ``fwd_bwd``, because the per-token weights are a function of the log
probs and cannot be known before the first pass. Whether that costs a second forward is a server-side choice:
retaining the graph between the two calls trades the extra forward for activation memory held across one round
trip. A co-located deployment pays neither, since the graph never leaves the process.
"""

from typing import Any

import torch
from trl.experimental.api import ForwardBackwardOutput


class ArcticTrainingClient:
    """Runs the model on an Arctic RL server while TRL keeps the loss.

    Configuration lives on this object rather than in TRL's config: the trainer takes the client as a
    constructed instance, so endpoint, temperature and loss name never become TRL config fields.

    Args:
        client: A :class:`~arctic_platform.client.client.SyncArcticRLClient`.
        temperature: Sampling temperature applied by the ``apply_temperature`` post-processor.
        loss_fn: Name of the server-side surrogate. ``pipeline._resolve_fn`` falls back to a dotted-path
            import, so ``"arctic_platform.integrations.trl.loss.weighted_logprob_sum"`` also resolves when the
            registry entry is not present. Unrelated to TRL's ``loss_fn`` argument, which is a callable and
            never leaves this process.
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

    def forward_backward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_fn: Any,
        aux_loss_coef: float = 0.0,
    ) -> ForwardBackwardOutput:
        # `model` is the trainer's local module. Arctic owns the weights, so it is unused here.
        #
        # AsyncGRPOTrainer packs a rank's samples into one padding-free row and marks sequence boundaries by
        # position_ids resets. The server wants padded [B, S] rows, so unpack first. This conversion is the
        # bulk of the adapter and is the main reason it belongs on this side of the wire.
        batch = _unpack_to_padded_rows(input_ids, position_ids, completion_mask)

        response = self.client.fwd_no_grad(
            batch,
            processing={
                "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                "loss_fn": None,
            },
        )
        out = response["batch"]

        # The server returns log probs under the roll(-1) convention and TRL expects the [:, 1:] shift, so the
        # repack aligns them rather than merely reshaping.
        log_probs = _repack_to_row(out["logprobs"], completion_mask)
        entropy = _repack_to_row(out["entropy"], completion_mask)

        # Evaluate the trainer's loss here, locally, on a leaf. This is the whole trick: the loss is a Python
        # callable in this process, so it never has to exist on the server or be named on the wire.
        leaf = log_probs.detach().requires_grad_(True)
        loss = loss_fn(leaf)
        (grad_log_probs,) = torch.autograd.grad(loss, leaf)

        def send_backward(grad_loss: torch.Tensor) -> None:
            # Fires from the trainer's `accelerator.backward`. `grad_loss` carries whatever scaling that
            # backward applies, so folding it in here keeps gradient accumulation correct.
            weights = _unpack_to_padded(grad_log_probs * grad_loss)
            payload = dict(batch)
            # `_shifted` marks the roll(-1) convention, matching the other log-prob tensors in the batch.
            payload["logprob_weights_shifted"] = weights
            self.client.fwd_bwd(
                payload,
                processing={"post": ["apply_temperature"], "loss_fn": self.loss_fn},
            )

        # Detached, because nothing on this side is connected to the model. The hook is what reaches it.
        reported_loss = loss.detach().requires_grad_(True)
        reported_loss.register_hook(send_backward)

        return ForwardBackwardOutput(
            loss=reported_loss,
            log_probs=log_probs.detach(),
            entropy=entropy,
            aux_loss=None,  # Arctic reports the MoE aux loss as a metric, not as a differentiable tensor
        )


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
# `old_log_probs`, no `advantages`, no `ref_log_prob`. TRL's `loss_fn` consumed those before the gradient was
# formed, so they never reach the wire.


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
