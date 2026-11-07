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

from arctic_platform.inference.vllm.glm5_batch_invariance import _single_split


def test_sparse_mla_attention_runs_one_split():
    seen = {}

    def flash_attn_varlen_func(*args, **kwargs):
        seen.update(kwargs)
        return "out"

    call = _single_split(flash_attn_varlen_func)
    assert call(q=1, num_splits=0) == "out"
    assert seen == {"q": 1, "num_splits": 1}
    assert call.__wrapped__ is flash_attn_varlen_func
