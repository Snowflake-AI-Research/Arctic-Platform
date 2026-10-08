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

from __future__ import annotations


def test_reference_liger_fused_ce_uses_fp32_accumulator(monkeypatch):
    import torch
    from liger_kernel.transformers import fused_linear_cross_entropy

    from arctic_platform.correctness.reference.hf_single_gpu import _build_liger_fused_cross_entropy

    captured = {}

    class FakeFusedCrossEntropy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        fused_linear_cross_entropy,
        "LigerFusedLinearCrossEntropyLoss",
        FakeFusedCrossEntropy,
    )

    _build_liger_fused_cross_entropy()

    assert captured == {
        "ignore_index": -100,
        "reduction": "mean",
        "accum_dtype": torch.float32,
    }
