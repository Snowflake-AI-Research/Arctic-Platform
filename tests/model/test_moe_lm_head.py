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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from arctic_platform.model.implementations.moe.layers.lm_head import FusedCrossEntropyOutputLinear


class _PerTokenLoss(nn.Module):
    def forward(self, weight, hidden_states, labels):
        assert weight.shape == (5, 3)
        assert hidden_states.shape == (6, 3)
        return labels.float() + 0.5


def test_fused_cross_entropy_can_return_target_logprobs():
    head = FusedCrossEntropyOutputLinear(3, 5)
    head.fused_logprobs = _PerTokenLoss()
    hidden_states = torch.randn(2, 3, 3)
    labels = torch.tensor([[0, 1, 2], [3, 4, -100]])

    output = head(hidden_states, labels=labels, dss_compute_logprobs=True)

    torch.testing.assert_close(output["logprobs"], -(labels.float() + 0.5))
    assert "loss" not in output
