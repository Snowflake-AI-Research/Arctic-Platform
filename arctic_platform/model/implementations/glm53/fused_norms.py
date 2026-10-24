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
"""Fused kernels for GLM-5.3-Flash's RMSNorms on CUDA.

The reference norms run every step as separate FP32 elementwise passes. On CUDA inputs:

- ``Glm5NextTextRMSNorm`` (layer, latent and final norms) runs Liger's kernel in ``llama`` casting
  mode: FP32 statistics, the normalized value cast to the input dtype, then the weight -- the
  reference's own order.
- ``Glm5NextTextRMSNormGated`` (the KDA output norm, ``weight * norm(x) * sigmoid(gate)`` in FP32)
  runs FLA's fused gated norm.

CPU inputs keep the reference forward.
"""

from __future__ import annotations

import types

from torch import Tensor
from torch import nn


def _rms_norm_forward(self, hidden_states: Tensor) -> Tensor:
    if not hidden_states.is_cuda:
        return type(self).forward(self, hidden_states)
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    return LigerRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon, 0.0, "llama", False)


def _rms_norm_gated_forward(self, hidden_states: Tensor, gate: Tensor | None = None) -> Tensor:
    if not hidden_states.is_cuda or gate is None:
        return type(self).forward(self, hidden_states, gate)
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated(hidden_states, gate, self.weight, None, activation="sigmoid", eps=self.variance_epsilon)


def fuse_glm5_next_norms(model: nn.Module) -> dict[str, int]:
    """Route every GLM-5.3-Flash RMSNorm and gated RMSNorm in ``model`` through the fused kernels."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNormGated

    counts = {"rms_norm": 0, "rms_norm_gated": 0}
    for module in model.modules():
        if type(module) is Glm5NextTextRMSNormGated and module.activation == "sigmoid":
            module.forward = types.MethodType(_rms_norm_gated_forward, module)
            counts["rms_norm_gated"] += 1
        elif type(module) is Glm5NextTextRMSNorm:
            module.forward = types.MethodType(_rms_norm_forward, module)
            counts["rms_norm"] += 1
    return counts
