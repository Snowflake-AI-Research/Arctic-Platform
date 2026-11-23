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

"""A verified broadcast load reports a digest of every parameter the rank serves afterwards."""

import hashlib

import torch

from arctic_platform.debug import weight_sync as evidence


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float8_e4m3fn).T)
        self.tied_weight = self.weight
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def load_weights(self, weights):
        for name, value in weights:
            getattr(self, name).data.copy_(value)


def _vllm_digest(value):
    """vLLM's ``compute_tensor_digests`` for a plain parameter."""
    return hashlib.sha256(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()).hexdigest()


def test_verification_reports_the_installed_digest_of_every_parameter():
    model = _Model()
    verifier = evidence.ShadowVerifier(model, max_shadow_bytes=1024)
    wire = torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2.5
    verifier.load("weight", wire)
    with torch.no_grad():
        model.weight.copy_(wire.flip(1))

    record = verifier.finish()

    assert record["installed"] == {
        "weight": _vllm_digest(wire.flip(1).to(model.weight.dtype)),
        "bias": _vllm_digest(model.bias),
    }
    assert record["changed_after_verification"] == ["weight"]
