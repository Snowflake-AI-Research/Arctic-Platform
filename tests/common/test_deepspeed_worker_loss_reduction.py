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

"""Objective-owned response reduction in the native DeepSpeed worker."""

from __future__ import annotations

import torch

from arctic_platform.rl.processors import prepare_request_loss


class _Engine:
    global_rank = 0

    def __init__(self):
        self.parameter = torch.tensor(-2.0, requires_grad=True)
        self.steps = 0

    def gradient_accumulation_steps(self):
        return 2

    def train(self):
        pass

    def __call__(self, input_ids, **_kwargs):
        return {"logprobs": self.parameter.expand_as(input_ids)}

    def backward(self, loss, scale_wrt_gas=False):
        assert scale_wrt_gas is False
        loss.backward()

    def step(self):
        self.steps += 1


def test_native_worker_sums_globally_normalized_gas_losses_without_rescaling_gradients():
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    request = {
        "batch": [
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "labels": torch.ones(1, 1, dtype=torch.long),
                "kd_mask": torch.ones(1, 1),
            },
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "labels": torch.ones(1, 1, dtype=torch.long),
                "kd_mask": torch.ones(1, 1),
            },
        ],
        "meta": {"pad_token_id": 0},
        "processing": {
            "loss_fn": "grouped_distillation",
            "config": {"kd_coef": 0.0},
        },
    }
    prepare_request_loss(request)
    request["meta"]["dp_size"] = 1

    worker_class = DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.rank = 0
    worker.world_size = 1
    worker.sp_size = 1
    worker.engine = _Engine()
    worker._device = torch.device("cpu")
    worker.cpu_device = torch.device("cpu")

    response = worker.forward_backward(request)

    assert request["processing"]["config"] == {
        "kd_coef": 0.0,
        "kd_batch_num_tokens": 2.0,
        "dp_size": None,
    }
    assert response["avg_loss"] == 2.0
    assert worker.engine.parameter.grad.item() == -1.0
    assert worker.engine.steps == 1
