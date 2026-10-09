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

"""Two-rank driver for the ragged tiled-MLP collective schedule regression."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from arctic_platform.model.implementations.gpu.tiled_mlp import apply_tiled_mlp


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 16, bias=False)
        self.up = nn.Linear(8, 16, bias=False)
        self.down = nn.Linear(16, 8, bias=False)

    def forward(self, hidden_states):
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


def main():
    dist.init_process_group("gloo")
    calls = 0

    def zero3_collective_forward(module, hidden_states):
        nonlocal calls
        # A ZeRO-3 projection gathers its partitioned parameters collectively. Unequal tile counts make peers
        # execute a different number of collectives; this all-reduce gives that mismatch the same failure mode.
        collective = torch.ones(1)
        dist.all_reduce(collective)
        calls += 1
        return module.down(F.silu(module.gate(hidden_states)) * module.up(hidden_states))

    try:
        model = MLP().double()
        for index, parameter in enumerate(model.parameters()):
            parameter.ds_id = index
        apply_tiled_mlp(
            model,
            is_target=lambda module: isinstance(module, MLP),
            mlp_forward=zero3_collective_forward,
            compute_params=lambda module: list(module.parameters()),
            token_chunk_size=4,
        )
        rank = int(os.environ["RANK"])
        tokens = 5 if rank == 0 else 2
        hidden_states = torch.randn(tokens, 8, dtype=torch.float64, requires_grad=True)
        model(hidden_states).square().sum().backward()
        assert calls == 4, (rank, calls)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
