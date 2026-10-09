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

"""PrimeRL tiled shared experts defer every trainable parameter added before the first forward."""

import torch
from torch import nn

from arctic_platform.model.implementations.gpu.tiled_mlp import trainable_parameters


class SharedExpertWithLora(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(4, 8, bias=False)
        self.w2 = nn.Linear(8, 4, bias=False)
        self.w3 = nn.Linear(4, 8, bias=False)
        self.lora_A = nn.Parameter(torch.randn(2, 4))
        self.frozen_adapter = nn.Parameter(torch.randn(2, 4), requires_grad=False)


def test_prime_shared_expert_discovers_lora_parameters_at_forward_time():
    expert = SharedExpertWithLora()

    deferred = trainable_parameters(expert)

    expected = [parameter for parameter in expert.parameters() if parameter.requires_grad]
    assert [id(parameter) for parameter in deferred] == [id(parameter) for parameter in expected]
    assert any(parameter is expert.lora_A for parameter in deferred)
    assert all(parameter is not expert.frozen_adapter for parameter in deferred)
