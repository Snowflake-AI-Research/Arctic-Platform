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

"""Context-parallel helpers used by DeepSeek V4 attention.

The default context is a single rank. The DeepSpeed loader installs the DSS
SP group on the model and every attention module when sequence parallel is on.
``gather_for_cp`` is the all-gather the compressor uses once that context exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn


@dataclass(frozen=True)
class CPContext:
    cp_group: dist.ProcessGroup | None = None
    cp_rank: int = 0
    cp_world_size: int = 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_world_size > 1


def gather_for_cp(t: torch.Tensor, cp_group: dist.ProcessGroup | None) -> torch.Tensor:
    if cp_group is None or dist.get_world_size(group=cp_group) == 1:
        return t
    return torch.cat(dist_nn.all_gather(t, group=cp_group), dim=1)
