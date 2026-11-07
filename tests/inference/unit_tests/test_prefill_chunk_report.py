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

import types

from arctic_platform.inference.vllm.prefill_chunk_report import chunk_report
from arctic_platform.inference.vllm.prefill_chunk_report import record_scheduled_chunk


def _request(computed: int, prompt: int):
    return types.SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt)


def test_records_prefill_chunks_from_the_cache_hit_and_ignores_decodes():
    request = _request(1568, 5000)  # resumed from a 1568-token cache hit
    for scheduled in (2048, 1384):
        record_scheduled_chunk(request, scheduled)
        request.num_computed_tokens += scheduled
    record_scheduled_chunk(request, 1)  # decode step
    assert chunk_report(request) == {"chunks": [[1568, 3616], [3616, 5000]], "preempted": False}


def test_a_restarted_prefill_is_reported_as_preempted():
    request = _request(0, 3000)
    record_scheduled_chunk(request, 2048)
    request.num_computed_tokens = 0
    record_scheduled_chunk(request, 3000)
    assert chunk_report(request)["preempted"] is True


def test_no_report_for_a_request_that_never_prefilled():
    assert chunk_report(_request(10, 5)) is None


def test_reports_the_first_attention_and_mamba_groups_blocks():
    from arctic_platform.inference.vllm.prefill_chunk_report import group_blocks

    class FullAttentionSpec:
        pass

    class MambaSpec:
        pass

    class GdnSpec(MambaSpec):
        pass

    groups = [types.SimpleNamespace(kv_cache_spec=spec()) for spec in (GdnSpec, FullAttentionSpec, GdnSpec)]
    assert group_blocks(groups, ([3, 0, 9], [4, 5, 6], [7, 0, 8])) == {
        "mamba_blocks": [3, 0, 9],
        "attention_blocks": [4, 5, 6],
    }
    request = _request(0, 10)
    record_scheduled_chunk(request, 10)
    request._arctic_blocks = {"attention_blocks": [1]}
    assert chunk_report(request) == {"chunks": [[0, 10]], "preempted": False, "attention_blocks": [1]}
