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

"""CPU-only async distillation trainer: compute and weight sync stay on Arctic.

TRL ``AsyncDistillationTrainer`` still ``from_pretrained``s a local student.
This trainer follows the same hooks (``rollout_worker``, ``weight_transfer``,
``training_client``, ``optimizers``) but never loads a GPU model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from arctic_platform.integrations.trl_distill.client import ArcticOPDOptimizer
from arctic_platform.integrations.trl_distill.client import ArcticOPDTrainingClient
from arctic_platform.integrations.trl_distill.config import ArcticAsyncDistillationConfig
from arctic_platform.integrations.trl_distill.jsd import generalized_jsd
from arctic_platform.integrations.trl_distill.rollout import ArcticOPDRolloutWorker
from arctic_platform.integrations.trl_distill.stub import RemoteStudentStub
from arctic_platform.integrations.trl_distill.weights import ArcticOPDWeightTransfer


def _as_prompt_batches(prompts: Sequence[list[int]], batch_size: int, steps: int) -> list[list[list[int]]]:
    if not prompts:
        raise ValueError("train_prompts is empty")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batches: list[list[list[int]]] = []
    n = len(prompts)
    for step in range(steps):
        start = (step * batch_size) % n
        batches.append([prompts[(start + offset) % n] for offset in range(batch_size)])
    return batches


class ArcticAsyncDistillationTrainer:
    """CPU driver for on-policy distillation against ``ArcticOPDClient``."""

    def __init__(
        self,
        client: Any,
        train_prompts: Sequence[list[int]],
        args: ArcticAsyncDistillationConfig | None = None,
        *,
        rollout_worker: ArcticOPDRolloutWorker | None = None,
        weight_transfer: ArcticOPDWeightTransfer | None = None,
        training_client: ArcticOPDTrainingClient | None = None,
        optimizers: tuple[Any, Any] | None = None,
    ) -> None:
        self.args = args or ArcticAsyncDistillationConfig()
        self.client = client
        self.train_prompts = list(train_prompts)
        self.model = RemoteStudentStub()
        if any(p.is_cuda for p in self.model.parameters()):
            raise RuntimeError("ArcticAsyncDistillationTrainer must stay CPU-only")

        self.rollout_worker = rollout_worker or ArcticOPDRolloutWorker(
            client,
            teacher_top_k=self.args.teacher_top_k,
            temperature=self.args.temperature,
            teacher_temperature=self.args.teacher_temperature,
            add_tail_bucket=self.args.add_tail_bucket,
        )
        self.weight_transfer = weight_transfer or ArcticOPDWeightTransfer(client)
        self.training_client = training_client or ArcticOPDTrainingClient(
            client, temperature=self.args.temperature, pad_token_id=self.args.pad_token_id
        )
        if optimizers is not None:
            self.optimizer = optimizers[0]
        else:
            self.optimizer = ArcticOPDOptimizer(client)
        self.state: dict[str, Any] = {"global_step": 0, "log_history": []}

    def _jsd_loss(
        self,
        logits_k: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor,
        *,
        student_logit_logsumexp: torch.Tensor,
    ) -> torch.Tensor:
        return generalized_jsd(
            logits_k,
            teacher,
            mask,
            beta=self.args.beta,
            add_tail_bucket=self.args.add_tail_bucket,
            student_logit_logsumexp=student_logit_logsumexp,
        )

    def train(self) -> dict[str, Any]:
        """Run generate → remote gather → CPU JSD → remote backward → step → sync."""
        self.weight_transfer.init_weight_transfer()
        self.rollout_worker.start()
        batches = _as_prompt_batches(self.train_prompts, self.args.batch_size, self.args.steps)
        try:
            for step, prompts in enumerate(batches, start=1):
                samples = self.rollout_worker.generate_and_score(
                    prompts, max_tokens=self.args.max_completion_length
                )
                output = self.training_client.forward_samples(
                    samples,
                    self._jsd_loss,
                    pad_token_id=self.args.pad_token_id,
                    max_seq_len=self.args.max_seq_len,
                )
                output.loss.backward()
                self.optimizer.step(self.args.learning_rate)
                self.optimizer.zero_grad()
                if step % self.args.weight_sync_steps == 0:
                    self.weight_transfer.send_weights(iter(()))
                    self.rollout_worker.update_model_version(step)
                self.state["global_step"] = step
                self.state["log_history"].append({"step": step, "loss": float(output.loss.detach())})
        finally:
            self.rollout_worker.stop()
            self.weight_transfer.destroy()
        return self.state


def create_arctic_async_distillation_trainer(
    client: Any,
    train_prompts: Sequence[list[int]],
    args: ArcticAsyncDistillationConfig | None = None,
) -> ArcticAsyncDistillationTrainer:
    return ArcticAsyncDistillationTrainer(client, train_prompts, args)
