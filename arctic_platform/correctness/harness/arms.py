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

"""The global batch sizes Test 1 runs, and how many microbatches each becomes.

GAS1 contains one variable-length sequence per configured GPU and is capped at 64K padded token slots in
aggregate so the single-GPU reference can execute it. GAS4 contains four times those rows and reuses the same
per-sequence cap. A config may lower either limit but cannot raise the correctness workload above the 64K GAS1
ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace
from typing import List

GAS1_TOKEN_CAP = 65_536


@dataclass(frozen=True)
class ArmDefinition:
    name: str
    global_batch_size: int
    max_seq_len: int

    @property
    def total_tokens(self) -> int:
        return self.global_batch_size * self.max_seq_len

    def microbatches(self, max_tokens_per_mb: int, dp_size: int = 1) -> int:
        """Maximum per-DP-shard model calls after distributing the global rows.

        A row is never split, so an arm whose single row exceeds the budget is still one model call on the
        shard that receives it. The global batch is balanced over data-parallel shards before packing.
        """
        rows_per_mb = max(1, max_tokens_per_mb // self.max_seq_len)
        global_calls = math.ceil(self.global_batch_size / rows_per_mb)
        return math.ceil(global_calls / dp_size)


# Both cases use one padded width so batch-size effects are not confounded with sequence-length effects.
# Active row lengths vary below that width, preserving sequence boundaries and padding coverage. GAS1 is the
# control; GAS4 changes only the number of rows.
GAS_VALUES = (1, 4)


def max_sequence_tokens(config_max_seq_len: int, gpu_count: int) -> int:
    """Largest padded row width that keeps the whole GAS1 batch within 64K slots."""
    if gpu_count < 1:
        raise ValueError(f"gpu_count must be positive, got {gpu_count}")
    if config_max_seq_len < 1:
        raise ValueError(f"max sequence length must be positive, got {config_max_seq_len}")
    return min(config_max_seq_len, GAS1_TOKEN_CAP // gpu_count)


def correctness_microbatch_tokens(configured_tokens: int) -> int:
    """Training configs may lower the correctness budget but cannot raise it above 64K."""
    if configured_tokens < 1:
        raise ValueError(f"microbatch token budget must be positive, got {configured_tokens}")
    return min(configured_tokens, GAS1_TOKEN_CAP)


def scaled_for_placement(arms: List[ArmDefinition], multiple: int) -> List[ArmDefinition]:
    """The reviewed cases with proportionally more rows, for a config placed on a multiple of its GPUs.

    A wider placement widens data parallelism, and a data-parallel shard that receives no row has nothing
    to reduce, so the row count has to grow with the width. Only the row count grows: the padded width of
    each sequence, the case names, and the seed are the reviewed ones, so the wider run differs from the
    declared-width run in how many sequences it carries and in nothing else.

    Each width therefore has its own single-GPU reference, because the reference executes the rows of the
    case it is compared against. A wider run costs one extra reference pass.
    """
    if multiple < 1:
        raise ValueError(f"placement multiple must be at least 1, got {multiple}")
    if multiple == 1:
        return list(arms)
    return [replace(arm, global_batch_size=arm.global_batch_size * multiple) for arm in arms]


def arms_for(max_row_tokens: int, gpu_count: int) -> List[ArmDefinition]:
    """One and four GAS groups, each with one variable-length row per configured GPU."""
    row_tokens = max_sequence_tokens(max_row_tokens, gpu_count)
    return [ArmDefinition(f"gas{gas}", gpu_count * gas, row_tokens) for gas in GAS_VALUES]
