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

"""Remote student train step for TRL async distillation.

TRL-side ``AsyncDistillationTrainer`` does not yet accept ``training_client=``.
This class is the Arctic half: gather student logits at the teacher's top-k
ids, let ``loss_fn`` (JSD) run in-process, ship ``dL/d(logits_k)`` as
``logit_weights``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from arctic_platform.integrations.trl_distill.batch_layout import samples_to_train_batch
from arctic_platform.integrations.trl_distill.types import RolloutSample

GATHER_POST = ["apply_temperature", "gather_logits_at_ids"]
SURROGATE_LOSS = "weighted_gathered_logit_sum"


@dataclass
class ForwardBackwardOutput:
    loss: torch.Tensor
    gathered_logits: torch.Tensor
    teacher_logprobs: torch.Tensor
    loss_mask: torch.Tensor


class ArcticOPDTrainingClient:
    """Arctic-hosted student, TRL-hosted JSD.

    ``loss_fn(gathered_logits, teacher_logprobs, loss_mask) -> scalar``.
    """

    def __init__(self, client: Any, *, temperature: float = 1.0, pad_token_id: int = 0) -> None:
        self.client = client
        self.temperature = temperature
        self._pad_token_id = pad_token_id
        self._bound_support: tuple[torch.Tensor, torch.Tensor | None] | None = None

    def _meta(self) -> dict:
        return {"temperature": self.temperature, "zorro_train_enable": False}

    def _processing(self, *, loss_fn: str | None) -> dict:
        return {"post": list(GATHER_POST), "loss_fn": loss_fn}

    def forward_samples(
        self,
        samples: list[RolloutSample],
        loss_fn: Callable[..., torch.Tensor],
        *,
        pad_token_id: int,
        max_seq_len: int | None = None,
    ) -> ForwardBackwardOutput:
        batch = samples_to_train_batch(samples, pad_token_id, max_seq_len=max_seq_len)
        response = self.client.fwd_no_grad(
            batch,
            processing=self._processing(loss_fn=None),
            meta=self._meta(),
        )
        out = response.get("batch", response)
        gathered = out.get("gathered_logits")
        if gathered is None:
            raise RuntimeError(
                "student fwd_no_grad returned no gathered_logits; "
                "server must run post=['apply_temperature', 'gather_logits_at_ids']"
            )
        if not torch.is_tensor(gathered):
            gathered = torch.as_tensor(gathered)
        teacher = batch["teacher_logprobs"].to(device=gathered.device, dtype=gathered.dtype)
        mask = batch["loss_mask"].to(device=gathered.device)
        leaf = gathered.detach().requires_grad_(True)
        loss = loss_fn(leaf, teacher, mask)
        (grad_logits,) = torch.autograd.grad(loss, leaf)

        def send_backward(grad_loss: torch.Tensor) -> None:
            weights = (grad_logits * grad_loss).detach()
            self.client.fwd_bwd(
                {**batch, "logit_weights": weights},
                processing=self._processing(loss_fn=SURROGATE_LOSS),
                meta=self._meta(),
            )

        reported = loss.detach().requires_grad_(True)
        reported.register_hook(send_backward)
        return ForwardBackwardOutput(
            loss=reported,
            gathered_logits=gathered.detach(),
            teacher_logprobs=teacher,
            loss_mask=mask,
        )

    def forward_backward(
        self,
        model: Any,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        aux_loss_coef: float = 0.0,
    ) -> Any:
        """TRL ``TrainingClientProtocol``: ignore ``model``, run student compute on Arctic.

        ``loss_fn`` is ``log_probs[B, T-1] -> scalar``. Teacher support must be
        bound first via :meth:`bind_teacher_support` for a gather path; otherwise
        sampled-token logprobs from ``compute_entropy_and_logprobs`` are used.
        """
        del model, aux_loss_coef
        if self._bound_support is not None:
            gather_ids, _teacher = self._bound_support
            batch = {
                "input_ids": input_ids,
                "attention_mask": (input_ids != self._pad_token_id).long(),
                "gather_token_ids": gather_ids,
            }
            response = self.client.fwd_no_grad(
                batch, processing=self._processing(loss_fn=None), meta=self._meta()
            )
            out = response.get("batch", response)
            gathered = out["gathered_logits"]
            if not torch.is_tensor(gathered):
                gathered = torch.as_tensor(gathered)
            leaf = torch.log_softmax(gathered.float(), dim=-1)
            # Protocol loss_fn wants [B, T-1]; use realized-token slice when 3D.
            log_probs = leaf[..., 0] if leaf.ndim == 3 else leaf
            if log_probs.shape[-1] == input_ids.shape[-1]:
                log_probs = log_probs[..., :-1]
            proto_leaf = log_probs.detach().requires_grad_(True)
            loss = loss_fn(proto_leaf)
            (grad_log,) = torch.autograd.grad(loss, proto_leaf)

            def send_backward(grad_loss: torch.Tensor) -> None:
                weights = gather_ids.new_zeros(gathered.shape, dtype=gathered.dtype)
                slice_w = (grad_log * grad_loss).detach()
                if weights.ndim == 3:
                    weights[..., :-1, 0] = slice_w
                else:
                    weights[..., :-1] = slice_w
                self.client.fwd_bwd(
                    {**batch, "logit_weights": weights},
                    processing=self._processing(loss_fn=SURROGATE_LOSS),
                    meta=self._meta(),
                )

            reported = loss.detach().requires_grad_(True)
            reported.register_hook(send_backward)
            entropy = torch.zeros_like(proto_leaf)
            return _protocol_output(reported, proto_leaf.detach(), entropy)

        batch = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self._pad_token_id).long(),
        }
        response = self.client.fwd_no_grad(
            {
                **batch,
            },
            processing={"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
            meta=self._meta(),
        )
        out = response.get("batch", response)
        log_probs = out["logprobs"]
        if not torch.is_tensor(log_probs):
            log_probs = torch.as_tensor(log_probs)
        if log_probs.shape[-1] == input_ids.shape[-1]:
            log_probs = log_probs[..., :-1]
        leaf = log_probs.detach().requires_grad_(True)
        loss = loss_fn(leaf)
        (grad_log,) = torch.autograd.grad(loss, leaf)

        def send_logprob_backward(grad_loss: torch.Tensor) -> None:
            weights = torch.zeros_like(out["logprobs"]) if torch.is_tensor(out.get("logprobs")) else None
            del weights
            self.client.fwd_bwd(
                {**batch, "logprob_weights_shifted": (grad_log * grad_loss).detach()},
                processing={
                    "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                    "loss_fn": "weighted_logprob_sum",
                },
                meta=self._meta(),
            )

        reported = loss.detach().requires_grad_(True)
        reported.register_hook(send_logprob_backward)
        entropy = out.get("entropy")
        if entropy is None:
            entropy = torch.zeros_like(leaf)
        elif torch.is_tensor(entropy) and entropy.shape[-1] == input_ids.shape[-1]:
            entropy = entropy[..., :-1]
        return _protocol_output(reported, leaf.detach(), entropy)

    def bind_teacher_support(
        self, gather_token_ids: torch.Tensor, teacher_logprobs: torch.Tensor | None = None
    ) -> None:
        self._bound_support = (gather_token_ids, teacher_logprobs)

    def clear_teacher_support(self) -> None:
        self._bound_support = None


def _protocol_output(loss: torch.Tensor, log_probs: torch.Tensor, entropy: torch.Tensor) -> Any:
    try:
        from trl.experimental.api import ForwardBackwardOutput as TRLOut
    except Exception:
        TRLOut = None
    if TRLOut is not None:
        return TRLOut(loss=loss, log_probs=log_probs, entropy=entropy, aux_loss=None)
    return ForwardBackwardOutput(
        loss=loss,
        gathered_logits=log_probs,
        teacher_logprobs=torch.zeros_like(log_probs),
        loss_mask=torch.ones(log_probs.shape[:2], dtype=torch.bool, device=log_probs.device),
    )


class ArcticOPDOptimizer:
    """Calls ``ArcticOPDClient.step``; clip/LR live in the server DeepSpeed config."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def step(self, learning_rate: float | None = None) -> dict:
        return self.client.step(learning_rate)

    def zero_grad(self) -> None:
        pass
