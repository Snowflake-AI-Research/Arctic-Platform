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

Student train, student vLLM, and teacher vLLM stay on disjoint GPUs
(``OnPremConfig.colocate=False``), matching TRL's three-server layout.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_non_colocated_student(client: Any) -> None:
    """Reject a student server that shares GPUs between train and sample."""
    backend = getattr(getattr(client, "config", None), "backend", None)
    if backend is None:
        return
    if bool(getattr(backend, "colocate", False)):
        raise ValueError(
            "ArcticAsyncDistillationTrainer requires a non-colocated student: "
            "DeepSpeed train, student vLLM, and teacher vLLM on disjoint GPUs "
            "(OnPremConfig.colocate=False), matching TRL AsyncDistillationTrainer"
        )


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
    """CPU driver for on-policy distillation against ``ArcticOPDClient``.

    The student must be launched with ``colocate=False`` so train, student
    sample, and teacher sample do not share GPUs.
    """

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
        _require_non_colocated_student(client)
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

    def train(self, after_step: Callable[..., None] | None = None) -> dict[str, Any]:
        """Run generate → remote gather → CPU JSD → remote backward → step → sync.

        ``repeat_batch=True`` generates and teacher-scores once, then replays
        that rollout so a flat JSD curve is an update-path bug, not on-policy
        drift. ``after_step(step, output, row)`` may add probe fields to ``row``.
        """
        self.weight_transfer.init_weight_transfer()
        self.rollout_worker.start()
        batches = _as_prompt_batches(self.train_prompts, self.args.batch_size, self.args.steps)
        frozen_samples = None
        try:
            for step, prompts in enumerate(batches, start=1):
                if frozen_samples is None or not self.args.repeat_batch:
                    samples = self.rollout_worker.generate_and_score(
                        prompts, max_tokens=self.args.max_completion_length
                    )
                    if self.args.repeat_batch:
                        frozen_samples = samples
                else:
                    samples = frozen_samples
                output = self.training_client.forward_samples(
                    samples,
                    self._jsd_loss,
                    pad_token_id=self.args.pad_token_id,
                    max_seq_len=self.args.max_seq_len,
                )
                output.loss.backward()
                step_out = self.optimizer.step(self.args.learning_rate)
                self.optimizer.zero_grad()
                sync_s = 0.0
                if step % self.args.weight_sync_steps == 0:
                    sync_started = time.monotonic()
                    self.weight_transfer.send_weights(iter(()))
                    sync_s = time.monotonic() - sync_started
                    self.rollout_worker.update_model_version(step)
                self.state["global_step"] = step
                row: dict[str, Any] = {
                    "step": step,
                    "loss": float(output.loss.detach()),
                    "jsd": float(output.loss.detach()),
                    "tokens": int(output.loss_mask.sum().item()),
                    "sync_s": sync_s,
                    "grad_norm": _metric_float((step_out or {}).get("metrics") or {}, "grad_norm"),
                    "lr": _metric_float((step_out or {}).get("metrics") or {}, "last_lr"),
                }
                if after_step is not None:
                    after_step(step, output, row)
                self.state["log_history"].append(row)
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
