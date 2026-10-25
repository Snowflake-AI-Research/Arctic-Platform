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
"""The fused MXFP4 -> BF16 kernel matches the table-lookup reference bit for bit."""

import pytest
import torch

from arctic_platform.model.implementations.deepseek_v4 import dequantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused dequantization runs on CUDA")


@pytest.mark.parametrize("shape", [(3, 64, 96), (5, 33, 160), (48, 128, 2560)])
def test_fused_mxfp4_matches_the_reference_bitwise(shape):
    experts, rows, cols = shape
    torch.manual_seed(0)
    packed = torch.randint(-128, 128, (experts, rows, cols // 2), dtype=torch.int8, device="cuda")
    # e8m0 scales are powers of two; cover small, unit and large ones.
    scale = torch.exp2(torch.randint(-20, 20, (experts, rows, cols // 32), device="cuda").float())
    reference = torch.cat([dequantize.dequantize_weight(packed[e:e + 1], scale[e:e + 1]) for e in range(experts)])
    fused = dequantize.dequantize_mxfp4(packed, scale)
    assert fused.dtype == torch.bfloat16 and fused.shape == (experts, rows, cols)
    assert torch.equal(fused.view(torch.int16), reference.view(torch.int16))  # signed zeros included
