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
import torch
import torch.nn.functional as F

from arctic_platform.model.implementations.qwen38 import host_table

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@requires_cuda
@pytest.mark.parametrize("dim", [160, 7, 256])
def test_gather_from_pinned_host_matches_embedding(dim):
    torch.manual_seed(0)
    weight = torch.randn(10_000, dim, dtype=torch.bfloat16, device="cuda")
    table = host_table.move_to_host(weight)
    ids = torch.randint(0, weight.shape[0], (3, 4096, 16), device="cuda")
    got = host_table.gather_rows(table, ids)
    assert got.device == ids.device and got.shape == (3, 4096, 16, dim)
    assert torch.equal(got, F.embedding(ids, weight))


@requires_cuda
def test_out_of_range_ids_are_refused():
    table = host_table.move_to_host(torch.zeros(8, 4, dtype=torch.bfloat16))
    with pytest.raises(IndexError):
        host_table.gather_rows(table, torch.tensor([0, 8], device="cuda"))
