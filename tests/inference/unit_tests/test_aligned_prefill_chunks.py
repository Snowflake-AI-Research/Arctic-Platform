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

import pytest

from arctic_platform.inference.vllm import aligned_prefill_chunks
from arctic_platform.inference.vllm.aligned_prefill_chunks import aligned_chunk_tokens


@pytest.mark.parametrize(
    ("start", "num_new_tokens", "prefill_end", "expected"),
    [
        (0, 5000, 10000, 4096),  # floored to the alignment
        (4096, 3000, 10000, 0),  # cannot reach 8192 this step: wait
        (4096, 4096, 10000, 4096),  # exactly one aligned chunk
        (8192, 1808, 10000, 1808),  # the prompt's last chunk is unchanged
        (8192, 900, 10000, 0),
        (10000, 1, 10000, 1),  # past the prompt: decode
        (0, 300, 300, 300),  # a prompt shorter than the alignment
    ],
)
def test_chunks_end_on_multiples_of_the_alignment(start, num_new_tokens, prefill_end, expected):
    assert aligned_chunk_tokens(start, num_new_tokens, prefill_end, 4096) == expected


def test_alignment_zero_leaves_the_chunk_alone():
    assert aligned_chunk_tokens(0, 5000, 10000, 0) == 5000


def test_the_env_var_is_read_as_a_chunk_size(monkeypatch):
    monkeypatch.setenv(aligned_prefill_chunks.ENV_VAR, "8192")
    assert aligned_prefill_chunks.prefill_chunk_alignment() == 8192
    monkeypatch.setenv(aligned_prefill_chunks.ENV_VAR, "-1")
    with pytest.raises(ValueError):
        aligned_prefill_chunks.prefill_chunk_alignment()
