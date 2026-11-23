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
"""The GLM-5.3 mHC remix, fused and eager FP32, adds its terms in the sampler's order."""

import subprocess
import sys
from pathlib import Path
from runpy import run_path

import pytest
import torch

import arctic_platform
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_equal


def _check_mix_order(mix_streams, device):
    # For output stream zero, (1 + 2**-24) - 1 rounds to zero. The first three channels cover each adjacent pair.
    small = 2.0**-22
    streams = torch.tensor(
        [[small, 0, 0, 1], [-4, small, 0, 2], [0, -4, small, 3], [0, 0, -4, 4]], dtype=torch.bfloat16, device=device
    )
    sublayer_out = torch.full((4,), 2.0, dtype=streams.dtype, device=device)
    post = torch.tensor([0.5, 1.0, 1.5, 1.75], device=device)
    comb = torch.tensor([[2, 4, 1, 1], [2, 2, 2, 2], [2, 1, 2, 3], [2, 1, 3, 2]], device=device) / 8
    reference = post[:, None] * sublayer_out
    for weights, stream in zip(comb, streams, strict=True):
        reference += weights[:, None] * stream
    actual = mix_streams(post[None, None], comb[None, None], sublayer_out[None, None], streams[None, None])
    torch_assert_equal(actual[0, 0], reference.to(streams.dtype))


class TestMixOrder(TestCasePlus):
    def test_interpreted_mix_order(self):
        pytest.importorskip("triton")
        # Triton's own reduction helpers are decorated when it is first imported.
        env = self.get_env()
        env["TRITON_INTERPRET"] = "1"
        subprocess.run([sys.executable, self.test_file_path], env=env, check=True)

    @require_torch_gpu
    def test_cuda_mix_order(self):
        from arctic_platform.model.implementations.glm53.hyper_connections import mix_streams

        _check_mix_order(mix_streams, "cuda")
        _check_first_product("cuda")

    def test_eager_mix_order(self):
        _check_first_product("cpu")


def _check_first_product(device):
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import _remix

    weights = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32, device=device)
    post = weights.expand(1, 16, -1)
    comb = torch.stack([weights.roll(i) for i in range(4)]).expand(1, 16, -1, -1)
    sublayer_out = torch.full((1, 16, 256), 1.125, dtype=torch.bfloat16, device=device)
    streams = torch.zeros((1, 16, 4, 256), dtype=torch.bfloat16, device=device)
    streams[:, :, 0] = -1.125
    # ATen's addcmul can be unfused on CPU. CUDA must produce the correctly rounded FMA.
    rounded = -1.125 * weights
    if device == "cpu":
        expected = rounded.addcmul(weights, torch.full_like(weights, 1.125))
    else:
        expected = 1.125 * weights.double() + rounded.double()
    actual = _remix(False, True, post, comb, sublayer_out, streams)
    torch_assert_equal(actual, expected.to(streams.dtype)[None, None, :, None].expand_as(actual))


if __name__ == "__main__":
    # Load the kernel directly so the CPU case can run without the model package's training dependencies.
    path = Path(arctic_platform.__file__).parent / "model/implementations/glm53/hyper_connections.py"
    _check_mix_order(run_path(str(path))["mix_streams"], "cpu")
