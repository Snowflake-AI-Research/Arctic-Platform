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

"""Remote student train step for TRL-style async distillation.

``forward_samples`` is what ``ArcticAsyncDistillationTrainer`` calls: gather
student logits, run JSD on CPU with full-vocab logsumexp, ship
``dL/d(gathered)`` and ``dL/d(logsumexp)``.

``forward_backward`` matches the distillation ``TrainingClientProtocol`` added
on the TRL side (teacher support in the call, not GRPO ``loss_fn(log_probs)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from arctic_platform.integrations.trl_distill.batch_layout import samples_to_train_batch
from arctic_platform.integrations.trl_distill.jsd import generalized_jsd
from arctic_platform.integrations.trl_distill.types import RolloutSample

GATHER_POST = ["apply_temperature", "gather_logits_at_ids"]
SURROGATE_LOSS = "weighted_gathered_logit_sum"


@dataclass
class ForwardBackwardOutput:
    loss: torch.Tensor
    gathered_logits: torch.Tensor
    teacher_logprobs: torch.Tensor
    loss_mask: torch.Tensor


@dataclass
class DistillationForwardBackwardOutput:
    """Same fields as TRL ``DistillationForwardBackwardOutput``."""

    loss: torch.Tensor
    jsd_sum: torch.Tensor
    entropy_sum: torch.Tensor
    teacher_entropy_sum: torch.Tensor
    per_teacher_stats: torch.Tensor
    forward_time_s: float = 0.0


class ArcticOPDTrainingClient:
    """Arctic-hosted student, trainer-process JSD."""

    def __init__(self, client: Any, *, temperature: float = 1.0, pad_token_id: int = 0) -> None:
        self.client = client
        self.temperature = temperature
        self._pad_token_id = pad_token_id

    def _meta(self) -> dict:
        return {
            "temperature": self.temperature,
            "zorro_train_enable": False,
            "pad_token_id": self._pad_token_id,
            "worker_return_tensors": True,
        }

    def _processing(self, *, loss_fn: str | None) -> dict:
        return {"post": list(GATHER_POST), "loss_fn": loss_fn}

    def _remote_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "loss_mask": batch["loss_mask"],
            "gather_token_ids": batch["gather_token_ids"],
            "prompts": batch["prompts"],
        }

    def _require_gather(self, response: dict) -> tuple[torch.Tensor, torch.Tensor]:
        out = response.get("batch", response)
        gathered = out.get("gathered_logits")
        lse = out.get("logit_logsumexp")
        if gathered is None or lse is None:
            raise RuntimeError(
                "student fwd_no_grad must return gathered_logits and logit_logsumexp; "
                "server post=['apply_temperature', 'gather_logits_at_ids']"
            )
        if not torch.is_tensor(gathered):
            gathered = torch.as_tensor(gathered)
        if not torch.is_tensor(lse):
            lse = torch.as_tensor(lse)
        return gathered, lse

    def forward_samples(
        self,
        samples: list[RolloutSample],
        loss_fn: Callable[..., torch.Tensor],
        *,
        pad_token_id: int,
        max_seq_len: int | None = None,
    ) -> ForwardBackwardOutput:
        batch = samples_to_train_batch(samples, pad_token_id, max_seq_len=max_seq_len)
        remote = self._remote_batch(batch)
        response = self.client.fwd_no_grad(
            remote,
            processing=self._processing(loss_fn=None),
            meta=self._meta(),
        )
        gathered, lse = self._require_gather(response)
        teacher = batch["teacher_logprobs"].to(device=gathered.device, dtype=gathered.dtype)
        mask = batch["loss_mask"].to(device=gathered.device)
        leaf_g = gathered.detach().requires_grad_(True)
        leaf_lse = lse.detach().requires_grad_(True)
        loss = loss_fn(leaf_g, teacher, mask, student_logit_logsumexp=leaf_lse)
        grad_g, grad_lse = torch.autograd.grad(loss, (leaf_g, leaf_lse), allow_unused=True)
        if grad_g is None:
            grad_g = torch.zeros_like(leaf_g)
        if grad_lse is None:
            grad_lse = torch.zeros_like(leaf_lse)

        def send_backward(grad_loss: torch.Tensor) -> None:
            self.client.fwd_bwd(
                {
                    **remote,
                    "logit_weights": (grad_g * grad_loss).detach(),
                    "logsumexp_weights": (grad_lse * grad_loss).detach(),
                },
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
        token_mask: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        candidate_mask: torch.Tensor,
        teacher_id_idx: torch.Tensor | None,
        *,
        beta: float,
        teacher_temperature: float,
        add_tail_bucket: bool,
        num_teachers: int,
        autocast: Any = None,
    ) -> DistillationForwardBackwardOutput:
        """Distillation ``TrainingClientProtocol``: ignore ``model``, compute on Arctic."""
        del model, teacher_temperature, autocast, teacher_id_idx
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, S], got {tuple(input_ids.shape)}")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids")
        seq_len = input_ids.shape[-1]
        support = int(target_ids.shape[-1])
        mask_1d = token_mask.reshape(-1).bool()
        if mask_1d.numel() != seq_len - 1:
            raise ValueError(f"token_mask length {mask_1d.numel()} != seq_len-1 {seq_len - 1}")
        if target_ids.shape[0] != int(mask_1d.sum().item()):
            raise ValueError("target_ids rows must equal the number of True token_mask positions")

        gather = torch.full((1, seq_len, support), -1, dtype=torch.long)
        teacher = torch.full((1, seq_len, support), float("-inf"))
        placed = torch.zeros((1, seq_len, support), dtype=torch.bool)
        loss_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        gather[0, : seq_len - 1][mask_1d] = target_ids.long()
        teacher[0, : seq_len - 1][mask_1d] = teacher_logprobs.to(dtype=teacher.dtype)
        placed[0, : seq_len - 1][mask_1d] = candidate_mask.bool()
        teacher = teacher.masked_fill(~placed, float("-inf"))
        loss_mask[0, : seq_len - 1] = mask_1d

        pos = position_ids.reshape(-1)
        starts = (pos == 0).nonzero(as_tuple=False).flatten()
        cu_seqlens = torch.cat([starts, pos.new_tensor([pos.numel()])]).long()
        first_prompt = max(int((~mask_1d).nonzero(as_tuple=False).flatten()[0].item()) + 1, 1)
        prompts = input_ids[:, :first_prompt].clone()
        remote = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "loss_mask": loss_mask,
            "gather_token_ids": gather,
            "prompts": prompts,
        }
        response = self.client.fwd_no_grad(
            remote, processing=self._processing(loss_fn=None), meta=self._meta()
        )
        gathered, lse = self._require_gather(response)
        leaf_g = gathered.detach().requires_grad_(True)
        leaf_lse = lse.detach().requires_grad_(True)
        loss = generalized_jsd(
            leaf_g,
            teacher.to(device=leaf_g.device, dtype=leaf_g.dtype),
            loss_mask.to(device=leaf_g.device),
            beta=beta,
            add_tail_bucket=add_tail_bucket,
            student_logit_logsumexp=leaf_lse,
        )
        grad_g, grad_lse = torch.autograd.grad(loss, (leaf_g, leaf_lse))

        def send_backward(grad_loss: torch.Tensor) -> None:
            self.client.fwd_bwd(
                {
                    **remote,
                    "logit_weights": (grad_g * grad_loss).detach(),
                    "logsumexp_weights": (grad_lse * grad_loss).detach(),
                },
                processing=self._processing(loss_fn=SURROGATE_LOSS),
                meta=self._meta(),
            )

        reported = loss.detach().requires_grad_(True)
        reported.register_hook(send_backward)
        zeros = reported.new_zeros(())
        per_teacher = reported.new_zeros(3 * max(num_teachers, 1))
        return DistillationForwardBackwardOutput(
            loss=reported,
            jsd_sum=loss.detach(),
            entropy_sum=zeros,
            teacher_entropy_sum=zeros,
            per_teacher_stats=per_teacher,
        )


class ArcticOPDOptimizer:
    """Calls ``ArcticOPDClient.step``; clip/LR live in the server DeepSpeed config."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def step(self, learning_rate: float | None = None) -> dict:
        return self.client.step(learning_rate)

    def zero_grad(self) -> None:
        pass
