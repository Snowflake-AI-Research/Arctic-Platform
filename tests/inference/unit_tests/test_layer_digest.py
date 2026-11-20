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

import torch

from arctic_platform.inference.vllm.layer_digest import row_digest


def test_row_digest_tells_rows_apart_by_their_bits():
    x = torch.randn(4, 3, 8, dtype=torch.bfloat16)
    y = x.clone()
    y.view(torch.int16)[2, 1, 5] += 1  # the next bf16 value
    a, b = row_digest(x, 4), row_digest(y, 4)
    assert a.dtype == torch.long and a.shape == (4,)
    assert torch.equal(a[[0, 1, 3]], b[[0, 1, 3]]) and a[2] != b[2]
    assert row_digest(x, 5) is None
    assert torch.equal(row_digest(x.float(), 4)[[0, 1, 3]], row_digest(y.float(), 4)[[0, 1, 3]])
