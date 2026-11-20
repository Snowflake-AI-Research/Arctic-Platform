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
"""The fused GLM-5.3-Flash mHC remix adds in vLLM ``mhc_post``'s FP32 order: ``post * y`` first, then each stream.

Runs the Triton kernel under ``TRITON_INTERPRET=1`` on CPU in a child process (the flag must be set before the
kernel module is imported).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import json
    import torch
    from arctic_platform.model.implementations.glm53.hyper_connections import mix_streams

    torch.manual_seed(0)
    tokens, hidden = 3, 40
    # BF16 values held in FP32 storage, so the kernel's output is its FP32 sum before any final cast.
    streams = torch.randn(1, tokens, 4, hidden).bfloat16().float()
    streams[0, 0] = torch.tensor([2.0**-22, -4.0, 0.0, 0.0])[:, None]  # products 2**-24, -1, 0, 0
    sublayer_out = torch.randn(1, tokens, hidden).bfloat16().float()
    sublayer_out[0, 0] = 1.0
    post = torch.rand(1, tokens, 4) * 2
    post[0, 0] = 1.0
    comb = torch.softmax(torch.randn(1, tokens, 4, 4), dim=2)
    comb[0, 0] = 0.25
    fused = mix_streams(post, comb, sublayer_out, streams)
    reference = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2)
    for weights, stream in zip(comb.unbind(-2), streams.unbind(-2)):
        # Product and sum rounded separately, as the interpreter does (a CPU addcmul_ may fuse them).
        reference = reference + weights.unsqueeze(-1) * stream.unsqueeze(-2)
    print(json.dumps({
        "cancellation_token": fused[0, 0, 0, 0].item(),
        "differing_elements": int((fused != reference).sum()),
    }))
    """
)


def test_fused_mix_adds_post_times_output_first_then_each_stream_in_order():
    pytest.importorskip("triton")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    child = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert result == {"cancellation_token": 0.0, "differing_elements": 0}  # (1 + 2**-24) - 1 rounds to 0 in FP32
