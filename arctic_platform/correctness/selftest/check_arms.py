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

"""Correctness cases cap the one-GPU GAS1 workload at 64K padded token slots."""

from arctic_platform.correctness.harness.arms import arms_for
from arctic_platform.correctness.harness.arms import correctness_microbatch_tokens


def test_eight_gpu_arms_cap_gas1_at_64k_total() -> None:
    arms = arms_for(max_row_tokens=262144, gpu_count=8)

    assert [(arm.name, arm.global_batch_size, arm.max_seq_len, arm.total_tokens) for arm in arms] == [
        ("gas1", 8, 8192, 65536),
        ("gas4", 32, 8192, 262144),
    ]


def test_four_gpu_arms_scale_rows_and_sequence_cap() -> None:
    arms = arms_for(max_row_tokens=32768, gpu_count=4)

    assert [(arm.name, arm.global_batch_size, arm.max_seq_len) for arm in arms] == [
        ("gas1", 4, 16384),
        ("gas4", 16, 16384),
    ]


def test_lower_sequence_limit_is_preserved() -> None:
    gas1, gas4 = arms_for(max_row_tokens=2048, gpu_count=8)

    assert (gas1.global_batch_size, gas1.max_seq_len, gas1.total_tokens) == (8, 2048, 16384)
    assert (gas4.global_batch_size, gas4.max_seq_len, gas4.total_tokens) == (32, 2048, 65536)


def test_microbatch_count_uses_the_capped_correctness_budget() -> None:
    gas1, gas4 = arms_for(max_row_tokens=262144, gpu_count=8)
    budget = correctness_microbatch_tokens(262144)

    assert budget == 65536
    assert gas1.microbatches(budget, dp_size=1) == 1
    assert gas4.microbatches(budget, dp_size=1) == 4


def test_smaller_microbatch_budget_is_preserved() -> None:
    assert correctness_microbatch_tokens(10240) == 10240
