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

"""Sequence-parallel world and head-exchange checks."""

from __future__ import annotations

import pytest

from arctic_platform.model.loader import ModelParallelismMetadata
from arctic_platform.model.sequence_parallel import validate_sequence_parallel


def _metadata(**overrides) -> ModelParallelismMetadata:
    fields = dict(
        num_attention_heads=16,
        num_key_value_heads=2,
        has_linear_attention=True,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
    )
    fields.update(overrides)
    return ModelParallelismMetadata(**fields)


def test_disabled_sequence_parallel_skips_the_other_checks():
    validate_sequence_parallel(1, 3, _metadata(num_attention_heads=1, linear_num_key_heads=None))


def test_sequence_parallel_must_divide_the_world():
    with pytest.raises(ValueError, match="must be divisible by sp_size"):
        validate_sequence_parallel(8, 12, _metadata())


def test_ulysses_requires_the_query_head_count_to_divide():
    with pytest.raises(ValueError, match="num_attention_heads"):
        validate_sequence_parallel(8, 8, _metadata(num_attention_heads=12))


def test_sequence_context_parallel_skips_head_exchange():
    validate_sequence_parallel(8, 8, _metadata(num_attention_heads=12), skip_head_exchange=True)


def test_gated_delta_net_value_heads_must_cover_the_key_heads():
    with pytest.raises(ValueError, match="divisible by linear_num_key_heads"):
        validate_sequence_parallel(8, 8, _metadata(linear_num_key_heads=16, linear_num_value_heads=24))
