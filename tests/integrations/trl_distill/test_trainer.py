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

from __future__ import annotations

import torch

from arctic_platform.integrations.trl_distill import ArcticAsyncDistillationConfig
from arctic_platform.integrations.trl_distill import ArcticAsyncDistillationTrainer
from arctic_platform.integrations.trl_distill import RemoteStudentStub
from arctic_platform.integrations.trl_distill.jsd import generalized_jsd


class FakeOPD:
    def __init__(self):
        self.fwd_bwd_calls = 0
        self.steps = 0
        self.synced = 0

    def generate(self, prompts, sampling_params):
        del sampling_params
        return [{"token_ids": [3, 4], "logprobs": [-0.4, -0.5]} for _ in prompts]

    def generate_teacher(self, prompts, sampling_params):
        assert sampling_params["prompt_logprobs"] == 2
        return [
            {
                "prompt_logprobs": [
                    None,
                    {2: -0.1},
                    {3: -0.2, 9: -1.0},
                    {4: -0.3, 8: -2.0},
                ]
            }
            for _ in prompts
        ]

    def fwd_no_grad(self, batch, processing=None, meta=None):
        del processing, meta
        gather = batch["gather_token_ids"]
        gathered = torch.zeros(*gather.shape)
        gathered[..., 0] = 0.5
        return {"batch": {"gathered_logits": gathered}}

    def fwd_bwd(self, batch, processing=None, meta=None):
        del batch, processing, meta
        self.fwd_bwd_calls += 1
        return {"metrics": {}}

    def step(self, learning_rate=None):
        del learning_rate
        self.steps += 1
        return {"metrics": {}}

    def sync_weights(self, cuda_ipc=None, low_memory=None):
        del cuda_ipc, low_memory
        self.synced += 1
        return {"ok": True}

    def reset_student_prefix_cache(self, drain=True, timeout_s=60.0, retry_interval_s=0.1):
        del drain, timeout_s, retry_interval_s
        return {"ok": True}


def test_stub_stays_on_cpu():
    stub = RemoteStudentStub()
    assert all(not p.is_cuda for p in stub.parameters())


def test_jsd_forward_kl_is_finite():
    logits = torch.zeros(1, 2, 2)
    teacher = torch.tensor([[[-0.2, -1.0], [-0.3, -2.0]]])
    mask = torch.tensor([[True, True]])
    loss = generalized_jsd(logits, teacher, mask, beta=0.0, add_tail_bucket=True)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_cpu_trainer_runs_remote_compute_and_sync():
    backend = FakeOPD()
    trainer = ArcticAsyncDistillationTrainer(
        backend,
        train_prompts=[[1, 2]],
        args=ArcticAsyncDistillationConfig(
            steps=2,
            batch_size=1,
            teacher_top_k=2,
            max_completion_length=4,
            weight_sync_steps=1,
        ),
    )
    assert all(not p.is_cuda for p in trainer.model.parameters())
    state = trainer.train()
    assert state["global_step"] == 2
    assert len(state["log_history"]) == 2
    assert backend.fwd_bwd_calls == 2
    assert backend.steps == 2
    assert backend.synced == 2
    assert trainer.rollout_worker.model_version == 2
    assert trainer.rollout_worker._started is False
